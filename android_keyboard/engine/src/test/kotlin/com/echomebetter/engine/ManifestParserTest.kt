package com.echomebetter.engine

import com.echomebetter.engine.manifest.ManifestException
import com.echomebetter.engine.manifest.ManifestParser
import com.echomebetter.engine.manifest.PromptSegment
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ManifestParserTest {
    private val source = Fixtures.text("tiny-echo/model.json")
    private val valid = Json.parseToJsonElement(source).jsonObject

    private fun with(key: String, value: JsonElement?): String =
        JsonObject(if (value == null) valid - key else valid + (key to value)).toString()

    private fun withGeneration(key: String, value: JsonElement): String =
        with("generation", JsonObject(valid.getValue("generation").jsonObject + (key to value)))

    private fun assertRejected(text: String, message: Regex) {
        val error = assertFailsWith<ManifestException> { ManifestParser.parseIndex(text) }
        assertTrue(message.containsMatchIn(error.message!!), "\"${error.message}\" does not match $message")
    }

    @Test
    fun `reads the published index`() {
        val index = ManifestParser.parseIndex(source)
        assertEquals(setOf(Style.PROFESSIONAL, Style.GRAMMAR), index.adapters.keys)
        assertEquals(listOf(PromptSegment.StyleToken, PromptSegment.Token("<text_to_rewrite>"), PromptSegment.Input, PromptSegment.Token("</text_to_rewrite>"), PromptSegment.Token("</s>")), index.base.prompt)
        assertEquals(index.base.files.sumOf { it.bytes }, index.base.sizeBytes)
        assertEquals(index.base.files.joinToString(":") { it.sha256 }, index.base.fingerprint)
    }

    @Test
    fun `the exempt token list and the URL exemption default to off and are read when present`() {
        val plain = ManifestParser.parseIndex(source).base.generation
        assertEquals(emptySet(), plain.noRepeatExemptTokenIds)
        assertEquals(false, plain.noRepeatExemptUrls)
        assertEquals(setOf(3, 4), ManifestParser.parseIndex(withGeneration("noRepeatExemptTokenIds", JsonArray(listOf(JsonPrimitive(3), JsonPrimitive(4))))).base.generation.noRepeatExemptTokenIds)
        assertEquals(true, ManifestParser.parseIndex(withGeneration("noRepeatExemptUrls", JsonPrimitive(true))).base.generation.noRepeatExemptUrls)
    }

    @Test
    fun `rejects malformed generation settings`() {
        assertRejected(withGeneration("noRepeatExemptTokenIds", JsonArray(listOf(JsonPrimitive(400)))), Regex("noRepeatExemptTokenIds"))
        assertRejected(withGeneration("noRepeatExemptTokenIds", JsonArray(listOf(JsonPrimitive(-1)))), Regex("noRepeatExemptTokenIds"))
        assertRejected(withGeneration("noRepeatExemptTokenIds", JsonPrimitive(3)), Regex("noRepeatExemptTokenIds"))
        assertRejected(withGeneration("noRepeatExemptUrls", JsonPrimitive("yes")), Regex("noRepeatExemptUrls"))
    }

    private fun prompt(vararg segments: JsonObject) = JsonArray(segments.toList())

    @Test
    fun `rejects a manifest the engine cannot run`() {
        val adapters = valid.getValue("adapters").jsonObject
        assertRejected(with("schemaVersion", JsonPrimitive(1)), Regex("schemaVersion 1 is not supported"))
        assertRejected(with("ioContract", JsonPrimitive("other/1")), Regex("ioContract .*other/1.* is not supported"))
        assertRejected(with("prompt", prompt(buildJsonObject { put("styleToken", true) }, buildJsonObject { put("token", "</s>") })), Regex("""exactly one \{"input": true\}"""))
        assertRejected(with("prompt", prompt(buildJsonObject { put("input", true) }, buildJsonObject { put("token", "</s>") })), Regex("""exactly one \{"styleToken": true\}"""))
        assertRejected(with("adapters", JsonObject(adapters + ("casual" to adapters.getValue("grammar")))), Regex("""adapters\.casual"""))
        assertRejected(with("adapters", JsonObject(emptyMap())), Regex("at least one adapter"))
        assertRejected(with("fallbackAdapter", JsonPrimitive("concise")), Regex("fallbackAdapter .*concise.* is not in adapters"))
        val escaping = buildJsonObject { put("path", "../x.onnx"); put("bytes", 1); put("sha256", "a".repeat(64)) }
        assertRejected(with("files", JsonObject(valid.getValue("files").jsonObject + ("encoder" to escaping))), Regex("relative path inside"))
        assertRejected(with("inputTransforms", JsonArray(listOf(JsonPrimitive("rot13")))), Regex("inputTransforms"))
        assertRejected(with("limits", null), Regex("limits must be an object"))
        assertRejected("not json", Regex("not valid JSON"))
    }

    @Test
    fun `adapter sources must be inside the model folder or pinned to one commit of a Thelivu repository`() {
        val adapters = valid.getValue("adapters").jsonObject
        val grammar = adapters.getValue("grammar").jsonObject
        fun withSource(value: String) = with("adapters", JsonObject(adapters + ("grammar" to JsonObject(grammar + ("source" to JsonPrimitive(value))))))
        val pinned = "https://huggingface.co/Thelivu/Thelivu-750M-Grammar/resolve/${"a".repeat(40)}/onnx/"
        assertEquals(pinned, ManifestParser.parseIndex(withSource(pinned)).adapters.getValue(Style.GRAMMAR).source)
        assertRejected(withSource("https://huggingface.co/Thelivu/Thelivu-750M-Grammar/resolve/main/onnx/"), Regex("adapters.grammar.source"))
        assertRejected(withSource("https://example.com/Thelivu/x/resolve/${"a".repeat(40)}/"), Regex("adapters.grammar.source"))
        assertRejected(withSource("../outside/"), Regex("adapters.grammar.source"))
    }

    @Test
    fun `an adapter file is checked against the base it was trained on and the size model json gives`() {
        val index = ManifestParser.parseIndex(source)
        val reference = index.adapters.getValue(Style.GRAMMAR)
        val text = Fixtures.text("tiny-echo/${reference.source}${reference.manifest.path}")
        val adapter = ManifestParser.parseAdapterFile(text, Style.GRAMMAR, reference, index.base)
        assertEquals("<style:grammar>", adapter.styleToken)
        assertEquals(reference.source + "encoder.onnx_adapter", adapter.encoder.path)

        val reject = { candidate: String, style: Style, message: Regex ->
            val error = assertFailsWith<ManifestException> { ManifestParser.parseAdapterFile(candidate, style, reference, index.base) }
            assertTrue(message.containsMatchIn(error.message!!), error.message)
        }
        reject(text, Style.PROFESSIONAL, Regex("it is for grammar, not professional"))
        reject(text.replace(index.base.fingerprint, "other"), Style.GRAMMAR, Regex("trained on another base model"))
        val bigger = reference.copy(sizeBytes = reference.sizeBytes + 1)
        val error = assertFailsWith<ManifestException> { ManifestParser.parseAdapterFile(text, Style.GRAMMAR, bigger, index.base) }
        assertTrue("add up to" in error.message!!)
    }
}
