package com.echomebetter.modelstore

import com.echomebetter.engine.Style
import com.echomebetter.engine.manifest.AdapterManifest
import com.echomebetter.engine.manifest.AdapterReference
import com.echomebetter.engine.manifest.ManifestException
import com.echomebetter.engine.manifest.ManifestParser
import com.echomebetter.engine.manifest.ModelFile
import com.echomebetter.engine.manifest.ModelIndex
import com.echomebetter.engine.manifest.ModelManifest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import kotlinx.serialization.json.put
import java.io.File

/** An installed model: its manifest and where each file is on disk. */
class InstalledModel(
    val index: ModelIndex,
    val manifest: ModelManifest,
    private val root: File,
) {
    val encoder: File get() = File(root, "base/${index.base.encoder.fileName}")
    val decoder: File get() = File(root, "base/${index.base.decoder.fileName}")
    val tokenizer: File get() = File(root, "base/${index.base.tokenizer.fileName}")

    fun adapterFiles(style: Style): Pair<File, File>? {
        val adapter = manifest.adapters[style] ?: return null
        val folder = File(root, "adapters/${style.id}")
        return File(folder, adapter.encoder.fileName) to File(folder, adapter.decoder.fileName)
    }

    val sizeBytes: Long get() = index.base.sizeBytes + manifest.adapters.values.sumOf { a -> a.files.sumOf { it.bytes } }
}

/** The file's name, without the folder its manifest places it in. */
internal val ModelFile.fileName: String get() = path.substringAfterLast('/')

/**
 * The installed files, under one folder (each folder flat: files keep their names, not the folders model.json puts
 * them in):
 *
 *     base/model.json, tokenizer.json, encoder.onnx, decoder.onnx
 *     adapters/<style>/adapter.json, encoder.onnx_adapter, decoder.onnx_adapter, reference.json
 *     staging/        files being downloaded
 *
 * A base is installed once `base/model.json` exists, and an adapter once its `reference.json` exists: each is written
 * last, after every file it names is in place, so a crash mid-install leaves nothing that looks installed. Reading
 * checks each file's size; hashes were checked when the files were downloaded.
 *
 * One writer at a time: the app serialises installs.
 */
class ModelStore(val root: File) {
    internal val baseDir = File(root, "base")
    internal val adaptersDir = File(root, "adapters")
    internal val stagingDir = File(root, "staging")

    internal fun adapterDir(style: Style) = File(adaptersDir, style.id)

    /** The installed model, or null when no complete base is installed. */
    fun installed(): InstalledModel? {
        val modelJson = File(baseDir, MODEL_JSON)
        if (!modelJson.isFile) return null
        val index = try {
            ManifestParser.parseIndex(modelJson.readText())
        } catch (_: ManifestException) {
            return null
        }
        if (index.base.files.any { !hasFile(File(baseDir, it.fileName), it) }) return null
        val adapters = Style.entries.mapNotNull { style -> installedAdapter(style, index)?.let { style to it } }.toMap()
        return InstalledModel(index, ModelManifest(index.base, adapters), root)
    }

    private fun installedAdapter(style: Style, index: ModelIndex): AdapterManifest? {
        val folder = adapterDir(style)
        val referenceFile = File(folder, REFERENCE_JSON)
        val adapterJson = File(folder, ADAPTER_JSON)
        if (!referenceFile.isFile || !adapterJson.isFile) return null
        return try {
            val reference = readReference(referenceFile.readText())
            if (!hasFile(adapterJson, reference.manifest)) return null
            val adapter = ManifestParser.parseAdapterFile(adapterJson.readText(), style, reference, index.base)
            adapter.takeIf { it.files.all { file -> hasFile(File(folder, file.fileName), file) } }
        } catch (_: ManifestException) {
            null
        } catch (_: IllegalArgumentException) {
            null
        }
    }

    /** [style]'s adapter, and any partial download of it. */
    fun removeAdapter(style: Style) {
        adapterDir(style).deleteRecursively()
        File(stagingDir, "adapters/${style.id}").deleteRecursively()
    }

    /** Everything: the base, every adapter and any partial download. */
    fun removeAll() {
        // The commit marker first, so an interrupted removal leaves nothing that looks installed.
        File(baseDir, MODEL_JSON).delete()
        root.listFiles()?.forEach { it.deleteRecursively() }
    }

    /** Free space where the files go. */
    fun freeBytes(): Long = root.apply { mkdirs() }.usableSpace

    internal fun commitBase(modelJson: String, staged: File) {
        File(baseDir, MODEL_JSON).delete()
        // Adapters belong to the base they were trained on; a new base starts without them.
        adaptersDir.deleteRecursively()
        baseDir.mkdirs()
        staged.listFiles()?.forEach { moveInto(it, baseDir) }
        File(baseDir, MODEL_JSON).writeText(modelJson)
        staged.deleteRecursively()
    }

    internal fun commitAdapter(style: Style, reference: AdapterReference, staged: File) {
        val folder = adapterDir(style)
        folder.deleteRecursively()
        folder.mkdirs()
        staged.listFiles()?.forEach { moveInto(it, folder) }
        File(folder, REFERENCE_JSON).writeText(writeReference(reference))
        staged.deleteRecursively()
    }

    private fun moveInto(file: File, folder: File) {
        val target = File(folder, file.name)
        target.delete()
        check(file.renameTo(target)) { "could not move ${file.name} into ${folder.path}" }
    }

    private fun hasFile(file: File, expected: ModelFile) = file.isFile && file.length() == expected.bytes

    internal companion object {
        const val MODEL_JSON = "model.json"
        const val ADAPTER_JSON = "adapter.json"
        const val REFERENCE_JSON = "reference.json"

        fun writeReference(reference: AdapterReference): String = buildJsonObject {
            put("source", reference.source)
            put("sizeBytes", reference.sizeBytes)
            put("manifest", buildJsonObject {
                put("path", reference.manifest.path)
                put("bytes", reference.manifest.bytes)
                put("sha256", reference.manifest.sha256)
            })
        }.toString()

        /** Our own record, so a malformed one is damage: [IllegalArgumentException], and the adapter counts as not installed. */
        fun readReference(text: String): AdapterReference =
            try {
                val json = Json.parseToJsonElement(text).jsonObject
                val manifest = json.getValue("manifest").jsonObject
                AdapterReference(
                    source = json.getValue("source").jsonPrimitive.content,
                    manifest = ModelFile(manifest.getValue("path").jsonPrimitive.content, manifest.getValue("bytes").jsonPrimitive.long, manifest.getValue("sha256").jsonPrimitive.content),
                    sizeBytes = json.getValue("sizeBytes").jsonPrimitive.long,
                )
            } catch (error: NoSuchElementException) {
                throw IllegalArgumentException("reference.json is incomplete", error)
            }
    }
}
