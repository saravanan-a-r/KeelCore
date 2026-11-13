package com.echomebetter.engine.tokenizer

import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction

/**
 * Token strings → text. Each decoder maps a list of token strings to a new list (HuggingFace's `decode_chain`);
 * the final list is concatenated. Thelivu's chain is Replace("▁" → " "), ByteFallback, Fuse.
 */
internal fun interface Decoder {
    fun decode(tokens: List<String>): List<String>
}

internal object Decoders {
    private val BYTE_TOKEN = Regex("^<0x([0-9A-Fa-f]{2})>$")

    fun replace(pattern: Regex, content: String) = Decoder { tokens -> tokens.map { it.replace(pattern) { content } } }

    val fuse = Decoder { tokens -> listOf(tokens.joinToString("")) }

    fun sequence(steps: List<Decoder>) = Decoder { tokens -> steps.fold(tokens) { current, step -> step.decode(current) } }

    /** Runs of `<0xNN>` tokens become the UTF-8 text they encode (or one U+FFFD per byte if invalid). */
    val byteFallback = Decoder { tokens ->
        val out = ArrayList<String>(tokens.size)
        val pending = ArrayList<Byte>()
        fun flush() {
            if (pending.isEmpty()) return
            val decoder = Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
            try {
                out.add(decoder.decode(ByteBuffer.wrap(pending.toByteArray())).toString())
            } catch (_: CharacterCodingException) {
                repeat(pending.size) { out.add("�") }
            }
            pending.clear()
        }
        for (token in tokens) {
            val match = BYTE_TOKEN.matchEntire(token)
            if (match != null) {
                pending.add(match.groupValues[1].toInt(16).toByte())
            } else {
                flush()
                out.add(token)
            }
        }
        flush()
        out
    }
}
