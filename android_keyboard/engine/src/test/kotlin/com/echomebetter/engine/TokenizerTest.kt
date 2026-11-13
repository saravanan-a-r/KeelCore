package com.echomebetter.engine

import com.echomebetter.engine.tokenizer.Tokenizer
import com.echomebetter.engine.tokenizer.UnsupportedTokenizerException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** The Kotlin tokenizer against HuggingFace `tokenizers` (Python), ID for ID, on the extension's probes. */
class TokenizerTest {
    private val tokenizerJson = Fixtures.text("tokenizers/byte-fallback.tokenizer.json")
    private val tokenizer = Tokenizer.fromJson(tokenizerJson)

    @Test
    fun `matches HuggingFace tokenizers on every probe`() {
        val probes = Json.parseToJsonElement(Fixtures.text("tokenizers/byte-fallback.expected.json")).jsonArray
        check(probes.isNotEmpty())
        for (probe in probes.map { it.jsonObject }) {
            val text = probe.getValue("text").jsonPrimitive.content
            val ids = probe.getValue("ids").jsonArray.map { it.jsonPrimitive.int }
            assertEquals(ids, tokenizer.encode(text), "ids of \"$text\"")
            assertEquals(probe.getValue("decoded").jsonPrimitive.content, tokenizer.decode(ids), "decoding of \"$text\"")
        }
    }

    @Test
    fun `an invalid byte run decodes to one replacement character per byte`() {
        val lead = tokenizer.tokenToId("<0xC3>") ?: error("fixture has no byte pieces")
        assertEquals("�", tokenizer.decode(listOf(lead)))
    }

    @Test
    fun `a tokenizer with a normaliser is refused`() {
        val withNormalizer = tokenizerJson.replaceFirst("\"normalizer\": null", "\"normalizer\": {\"type\": \"NFKC\"}")
        check(withNormalizer != tokenizerJson) { "fixture layout changed" }
        assertFailsWith<UnsupportedTokenizerException> { Tokenizer.fromJson(withNormalizer) }
    }
}
