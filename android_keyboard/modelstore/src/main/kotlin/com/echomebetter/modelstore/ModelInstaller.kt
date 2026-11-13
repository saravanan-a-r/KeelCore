package com.echomebetter.modelstore

import com.echomebetter.engine.Style
import com.echomebetter.engine.manifest.ManifestParser
import com.echomebetter.engine.manifest.ModelFile
import com.echomebetter.engine.manifest.ModelIndex
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** How far an install is, in bytes. */
data class InstallProgress(val doneBytes: Long, val totalBytes: Long)

/**
 * Installs the published model into a [ModelStore]: the base (model.json, tokenizer, encoder, decoder) and, on top of
 * it, one adapter per style. [modelUrl] is the folder model.json is published in; adapters live in their own
 * repositories, pinned by commit in model.json, and every file is checked against the size and hash model.json (or
 * the adapter.json it vouches for) gives.
 */
class ModelInstaller(
    private val store: ModelStore,
    private val modelUrl: String,
    private val downloader: Downloader,
) {
    init {
        require(modelUrl.endsWith("/")) { "modelUrl must be a folder URL ending in /" }
    }

    /** The published model.json, parsed: what is on offer and how big it is. */
    suspend fun fetchIndex(): ModelIndex = ManifestParser.parseIndex(fetch(modelUrl + ModelStore.MODEL_JSON).decodeToString())

    /** Installs the published base, replacing an installed base that differs (and with it that base's adapters). */
    suspend fun installBase(onProgress: (InstallProgress) -> Unit): ModelIndex {
        val modelJson = fetch(modelUrl + ModelStore.MODEL_JSON).decodeToString()
        val index = ManifestParser.parseIndex(modelJson)
        val names = index.base.files.map { it.fileName }
        if (names.toSet().size != names.size) throw ModelStoreException.Integrity("model.json names two base files the same: $names")
        val installed = store.installed()
        if (installed != null && installed.index.base.fingerprint == index.base.fingerprint) return installed.index
        val staged = File(store.stagingDir, "base")
        downloadAll(index.base.files.map { it.copy(path = it.fileName) to modelUrl + it.path }, staged, onProgress)
        withContext(Dispatchers.IO) { store.commitBase(modelJson, staged) }
        return index
    }

    /** Installs [style]'s adapter for the installed base, as the published model.json lists it. */
    suspend fun installAdapter(style: Style, onProgress: (InstallProgress) -> Unit) {
        val installed = store.installed() ?: throw ModelStoreException.BaseMissing()
        val published = fetchIndex()
        if (published.base.fingerprint != installed.index.base.fingerprint) throw ModelStoreException.BaseChanged()
        val reference = published.adapters[style] ?: throw ModelStoreException.NoSuchAdapter(style.id)
        val adapterJsonBytes = fetch(urlOf(reference.manifest, reference.source))
        if (adapterJsonBytes.size.toLong() != reference.manifest.bytes || sha256(adapterJsonBytes) != reference.manifest.sha256) {
            throw ModelStoreException.Integrity("${style.id} adapter.json does not match model.json")
        }
        val adapter = ManifestParser.parseAdapterFile(adapterJsonBytes.decodeToString(), style, reference, installed.index.base)
        if (adapter.encoder.fileName == adapter.decoder.fileName) throw ModelStoreException.Integrity("${style.id} adapter.json names both halves the same")
        val staged = File(store.stagingDir, "adapters/${style.id}")
        downloadAll(adapter.files.map { it.copy(path = it.fileName) to urlOf(it, null) }, staged, onProgress)
        withContext(Dispatchers.IO) {
            File(staged, ModelStore.ADAPTER_JSON).writeBytes(adapterJsonBytes)
            store.commitAdapter(style, reference, staged)
        }
    }

    /** A file's URL: under its pinned source when it has one, else under [relativeTo] or the model's folder. */
    private fun urlOf(file: ModelFile, relativeTo: String?): String =
        when {
            file.source != null -> file.source + file.path
            relativeTo != null && ManifestParser.isPinnedSource(relativeTo) -> relativeTo + file.path
            relativeTo != null -> modelUrl + relativeTo + file.path
            else -> modelUrl + file.path
        }

    private suspend fun downloadAll(files: List<Pair<ModelFile, String>>, folder: File, onProgress: (InstallProgress) -> Unit) {
        val total = files.sumOf { it.first.bytes }
        val needed = files.sumOf { (file, _) ->
            val done = File(folder, file.fileName).takeIf { it.isFile }?.length() ?: File(folder, file.fileName + ".part").takeIf { it.isFile }?.length() ?: 0L
            file.bytes - done
        }
        val free = store.freeBytes()
        if (needed > free) throw ModelStoreException.InsufficientStorage(needed, free)
        var done = 0L
        onProgress(InstallProgress(0, total))
        for ((file, url) in files) {
            downloader.download(url, file, File(folder, file.fileName)) { bytes ->
                done += bytes
                onProgress(InstallProgress(done, total))
            }
        }
    }

    /** A small file (model.json, adapter.json), whole. */
    private suspend fun fetch(url: String): ByteArray = withContext(Dispatchers.IO) {
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.instanceFollowRedirects = true
            connection.setRequestProperty("User-Agent", downloader.userAgent)
            val status = connection.responseCode
            if (status != HttpURLConnection.HTTP_OK) throw ModelStoreException.Network("$url: the server answered $status")
            connection.inputStream.use { it.readBytes() }
        } catch (error: IOException) {
            throw ModelStoreException.Network("could not fetch $url: ${error.message}", error)
        } finally {
            connection.disconnect()
        }
    }

    private fun sha256(bytes: ByteArray): String =
        java.security.MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}
