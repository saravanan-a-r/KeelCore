package com.echomebetter.engine.tokenizer

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive

class UnsupportedTokenizerException(component: String, type: String) : Exception("unsupported tokenizer $component: $type")

/**
 * Thelivu's tokenizer, built from its `tokenizer.json`.
 *
 * The format is the one tokenizer/training/export_hf.py writes from the SentencePiece model: no normaliser; a
 * Metaspace pre-tokenizer that turns spaces into "▁" without adding a prefix or splitting; a Unigram model with byte
 * fallback; no added tokens and no post-processor. Control pieces (`</s>`, the style and frame markers) are ordinary
 * vocabulary pieces, so text that spells one encodes to it; the prompt builder refuses such user text.
 *
 * A tokenizer.json in any other format is refused when it is loaded: a step silently skipped would change token IDs
 * without any error, and the model would quietly degrade.
 */
class Tokenizer private constructor(json: JsonObject) {
    private val replacement: String
    private val decoder: Decoder?
    private val model: UnigramModel

    init {
        val modelJson = json["model"] as? JsonObject
        val modelType = modelJson?.get("type")?.let { (it as? JsonPrimitive)?.content }
        if (modelJson == null || modelType != "Unigram") throw UnsupportedTokenizerException("model", modelType.toString())
        typeOf(json["normalizer"])?.let { throw UnsupportedTokenizerException("normalizer", it) }
        typeOf(json["post_processor"])?.let { throw UnsupportedTokenizerException("post_processor", it) }
        val added = json["added_tokens"] as? JsonArray
        if (added != null && added.isNotEmpty()) throw UnsupportedTokenizerException("added_tokens", "${added.size} tokens")
        replacement = metaspaceReplacement(json["pre_tokenizer"])
        decoder = buildDecoder(json["decoder"])
        model = UnigramModel(
            pieces = modelJson.getValue("vocab").jsonArray.map { entry ->
                val pair = entry.jsonArray
                pair[0].jsonPrimitive.content to pair[1].jsonPrimitive.double
            },
            unkId = (modelJson["unk_id"] as? JsonPrimitive)?.intOrNull,
            byteFallback = (modelJson["byte_fallback"] as? JsonPrimitive)?.booleanOrNull ?: false,
        )
    }

    /** The id of a vocabulary piece, or null. */
    fun tokenToId(token: String): Int? = model.tokenToId(token)

    /** The vocabulary piece of an id (`▁` marks a space), or null. */
    fun piece(id: Int): String? = model.idToToken(id)

    fun encode(text: String): List<Int> = model.tokenize(text.replace(" ", replacement))

    fun decode(ids: List<Int>): String {
        val tokens = ids.mapNotNull { model.idToToken(it) }
        return decoder?.decode(tokens)?.joinToString("") ?: tokens.joinToString("")
    }

    companion object {
        fun fromJson(text: String): Tokenizer {
            val json = Json.parseToJsonElement(text) as? JsonObject ?: throw IllegalArgumentException("tokenizer.json is not an object")
            return Tokenizer(json)
        }

        private fun typeOf(value: JsonElement?): String? =
            if (value == null || value is JsonNull) null else ((value as? JsonObject)?.get("type") as? JsonPrimitive)?.content ?: value.toString()

        private fun metaspaceReplacement(value: JsonElement?): String {
            val json = value as? JsonObject
            val type = (json?.get("type") as? JsonPrimitive)?.content
            if (json == null || type != "Metaspace") throw UnsupportedTokenizerException("pre_tokenizer", type.toString())
            val addPrefixSpace = (json["add_prefix_space"] as? JsonPrimitive)?.booleanOrNull
            val scheme = (json["prepend_scheme"] as? JsonPrimitive)?.content ?: if (addPrefixSpace == false) "never" else "always"
            if (scheme != "never") throw UnsupportedTokenizerException("pre_tokenizer", "Metaspace with prepend_scheme $scheme")
            if ((json["split"] as? JsonPrimitive)?.booleanOrNull != false) throw UnsupportedTokenizerException("pre_tokenizer", "Metaspace with split")
            return (json["replacement"] as? JsonPrimitive)?.content ?: throw UnsupportedTokenizerException("pre_tokenizer", "Metaspace without replacement")
        }

        private fun buildDecoder(value: JsonElement?): Decoder? {
            if (value == null || value is JsonNull) return null
            val json = value as? JsonObject ?: throw UnsupportedTokenizerException("decoder", value.toString())
            return when (val type = (json["type"] as? JsonPrimitive)?.content) {
                "ByteFallback" -> Decoders.byteFallback
                "Fuse" -> Decoders.fuse
                "Replace" -> Decoders.replace(pattern(json["pattern"]), (json["content"] as JsonPrimitive).content)
                "Sequence" -> Decoders.sequence(json.getValue("decoders").jsonArray.mapNotNull(::buildDecoder))
                else -> throw UnsupportedTokenizerException("decoder", type.toString())
            }
        }

        private fun pattern(value: JsonElement?): Regex {
            val json = value as? JsonObject ?: throw UnsupportedTokenizerException("decoder", "Replace without a pattern")
            (json["String"] as? JsonPrimitive)?.let { return Regex(Regex.escape(it.content)) }
            (json["Regex"] as? JsonPrimitive)?.let { return Regex(it.content) }
            throw UnsupportedTokenizerException("decoder", "Replace pattern $json")
        }
    }
}
