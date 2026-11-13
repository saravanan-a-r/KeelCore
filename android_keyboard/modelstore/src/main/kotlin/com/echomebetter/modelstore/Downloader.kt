package com.echomebetter.modelstore

import com.echomebetter.engine.manifest.ModelFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

/**
 * Downloads one file to a target path, verified against the size and SHA-256 its manifest gives.
 *
 * The bytes go to `<target>.part` first. An interrupted download resumes from the bytes already there (an HTTP range
 * request); the complete file is hashed, and only a file with the right size and hash is renamed to the target, so
 * the target is either absent or exactly right. A transient failure is retried a few times, with
 * back-off, before it is reported.
 */
class Downloader(
    val userAgent: String,
    private val connectTimeoutMs: Int = 15_000,
    private val readTimeoutMs: Int = 30_000,
    private val attempts: Int = 4,
    private val retryDelayMs: Long = 2_000,
) {
    /** [onBytes] is called with the number of new bytes each time more arrive (on the downloading thread). */
    suspend fun download(url: String, expected: ModelFile, target: File, onBytes: (Long) -> Unit) = withContext(Dispatchers.IO) {
        if (target.isFile && target.length() == expected.bytes) {
            // Verified when it was written; renamed into place only after that.
            onBytes(expected.bytes)
            return@withContext
        }
        target.parentFile?.mkdirs()
        val part = File(target.path + ".part")
        var lastError: ModelStoreException? = null
        for (attempt in 1..attempts) {
            coroutineContext.ensureActive()
            try {
                fetch(url, expected, part, onBytes)
                verify(part, expected)
                if (!part.renameTo(target)) throw IOException("could not move ${part.name} into place")
                return@withContext
            } catch (error: ModelStoreException.Integrity) {
                part.delete()
                throw error
            } catch (error: ModelStoreException.Network) {
                lastError = error
            } catch (error: IOException) {
                lastError = ModelStoreException.Network("download of ${expected.path} failed: ${error.message}", error)
            }
            if (attempt < attempts) delay(retryDelayMs * attempt)
        }
        throw lastError!!
    }

    private suspend fun fetch(url: String, expected: ModelFile, part: File, onBytes: (Long) -> Unit) {
        var have = if (part.isFile) part.length() else 0L
        if (have > expected.bytes) {
            part.delete()
            have = 0
        }
        if (have == expected.bytes) {
            onBytes(have)
            return
        }
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = connectTimeoutMs
            connection.readTimeout = readTimeoutMs
            connection.instanceFollowRedirects = true
            connection.setRequestProperty("User-Agent", userAgent)
            if (have > 0) connection.setRequestProperty("Range", "bytes=$have-")
            val status = connection.responseCode
            val append = when {
                status == HttpURLConnection.HTTP_PARTIAL && have > 0 -> true
                status == HttpURLConnection.HTTP_OK -> false // no range support: start over
                status == 416 && have > 0 -> {
                    // The server has nothing past what is on disk: start over rather than trust it.
                    part.delete()
                    throw ModelStoreException.Network("the server refused to resume ${expected.path}")
                }
                else -> throw ModelStoreException.Network("${expected.path}: the server answered $status")
            }
            if (append) onBytes(have) else have = 0
            connection.inputStream.use { input ->
                FileOutputStream(part, append).use { output ->
                    val buffer = ByteArray(BUFFER_BYTES)
                    while (true) {
                        coroutineContext.ensureActive()
                        val read = input.read(buffer)
                        if (read < 0) break
                        if (have + read > expected.bytes) throw ModelStoreException.Integrity("${expected.path} is larger than ${expected.bytes} bytes")
                        output.write(buffer, 0, read)
                        have += read
                        onBytes(read.toLong())
                    }
                }
            }
            if (have != expected.bytes) throw ModelStoreException.Network("${expected.path} ended after $have of ${expected.bytes} bytes")
        } finally {
            connection.disconnect()
        }
    }

    private suspend fun verify(file: File, expected: ModelFile) {
        if (file.length() != expected.bytes) throw ModelStoreException.Integrity("${expected.path} is ${file.length()} bytes, expected ${expected.bytes}")
        val actual = sha256(file)
        if (actual != expected.sha256) throw ModelStoreException.Integrity("${expected.path} has the wrong SHA-256 ($actual)")
    }

    companion object {
        private const val BUFFER_BYTES = 256 * 1024

        suspend fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(BUFFER_BYTES)
                while (true) {
                    coroutineContext.ensureActive()
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
