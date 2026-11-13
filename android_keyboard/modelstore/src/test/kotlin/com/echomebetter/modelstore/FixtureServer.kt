package com.echomebetter.modelstore

import com.sun.net.httpserver.HttpServer
import java.io.File
import java.net.InetSocketAddress
import java.util.Collections

/**
 * Serves a folder over HTTP on localhost, with range requests, and with faults on request: a response cut off
 * after some bytes, or a file served with one byte changed.
 */
class FixtureServer(private val folder: File) : AutoCloseable {
    private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)

    /** Each request: its path and its Range header. */
    val requests: MutableList<Pair<String, String?>> = Collections.synchronizedList(ArrayList())

    /** Path → how many bytes to send before dropping the connection, for that path's next request only. */
    val cutAfter: MutableMap<String, Int> = Collections.synchronizedMap(HashMap())

    /** Paths served with their first byte changed. */
    val corrupt: MutableSet<String> = Collections.synchronizedSet(HashSet())

    /** Path → replacement content. */
    val overrides: MutableMap<String, ByteArray> = Collections.synchronizedMap(HashMap())

    val url: String get() = "http://127.0.0.1:${server.address.port}/"

    init {
        server.createContext("/") { exchange ->
            val path = exchange.requestURI.path.removePrefix("/")
            val range = exchange.requestHeaders.getFirst("Range")
            requests.add(path to range)
            val file = File(folder, path)
            val bytes = overrides[path] ?: if (file.isFile) file.readBytes() else null
            if (bytes == null) {
                exchange.sendResponseHeaders(404, -1)
                exchange.close()
                return@createContext
            }
            if (path in corrupt) bytes[0] = (bytes[0] + 1).toByte()
            val start = range?.removePrefix("bytes=")?.removeSuffix("-")?.toInt() ?: 0
            if (start >= bytes.size && range != null) {
                exchange.sendResponseHeaders(416, -1)
                exchange.close()
                return@createContext
            }
            val body = bytes.copyOfRange(start, bytes.size)
            if (range != null) exchange.responseHeaders.add("Content-Range", "bytes $start-${bytes.size - 1}/${bytes.size}")
            exchange.sendResponseHeaders(if (range != null) 206 else 200, body.size.toLong())
            val cut = cutAfter.remove(path)
            exchange.responseBody.use { out ->
                if (cut != null) {
                    out.write(body, 0, minOf(cut, body.size))
                    out.flush()
                    // Ending the exchange short of the declared length: the client sees a premature end of stream.
                    throw java.io.IOException("cut off on purpose")
                }
                out.write(body)
            }
        }
        server.start()
    }

    override fun close() = server.stop(0)
}
