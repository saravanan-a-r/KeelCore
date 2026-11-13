package com.echomebetter.engine.manifest

import com.echomebetter.engine.Style
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.Json as KotlinJson

/**
 * Validates model.json and adapter.json in full before anything is downloaded or loaded, so a mismatched or
 * hand-edited model fails with a precise message instead of producing wrong text later. A port of the extension's
 * `parseModelIndex` and `parseAdapterFile`; the rules are the same.
 */
object ManifestParser {
    const val SUPPORTED_SCHEMA_VERSION = 3
    const val ADAPTER_FILE_SCHEMA_VERSION = 1
    const val SUPPORTED_IO_CONTRACT = "cross-kv/1"

    /** The only place outside the model's folder files may come from: one commit, so what it names cannot change. */
    private val PINNED_SOURCE = Regex("""^https://huggingface\.co/Thelivu/[A-Za-z0-9][A-Za-z0-9._-]*/resolve/[0-9a-f]{40}/(?:[A-Za-z0-9._-]+/)*$""")
    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val URL_SCHEME = Regex("^[a-z][a-z0-9+.-]*:", RegexOption.IGNORE_CASE)

    fun parseIndex(text: String): ModelIndex {
        val json = Json.obj(parse(text), "manifest")
        checkSchema(json)
        val base = parseBase(json)
        val adapters = adapterReferences(json["adapters"])
        val fallback = Json.optional(json["fallbackAdapter"])
        if (fallback != null) {
            val style = (fallback as? JsonPrimitive)?.takeIf { it.isString }?.content?.let(Style::fromId)
            if (style == null || style !in adapters) throw ManifestException("fallbackAdapter $fallback is not in adapters")
        }
        return ModelIndex(base, adapters)
    }

    /**
     * An adapter.json, checked against the model that refers to it. Its file paths are relative to its own folder;
     * the result's are placed under [reference]'s source.
     */
    fun parseAdapterFile(text: String, style: Style, reference: AdapterReference, base: BaseModel): AdapterManifest {
        val json = Json.obj(parse(text), "adapter.json")
        val version = (json["schemaVersion"] as? JsonPrimitive)?.content
        if (version != ADAPTER_FILE_SCHEMA_VERSION.toString()) {
            throw ManifestException("adapter.json: schemaVersion $version is not supported (expected $ADAPTER_FILE_SCHEMA_VERSION)")
        }
        val declared = (json["style"] as? JsonPrimitive)?.content
        if (declared != style.id) throw ManifestException("adapter.json: it is for $declared, not ${style.id}")
        if ((json["baseFingerprint"] as? JsonPrimitive)?.content != base.fingerprint) {
            throw ManifestException("adapter.json: the ${style.id} adapter was trained on another base model")
        }
        val files = Json.obj(json["files"], "adapter.json files")
        val pinned = isPinnedSource(reference.source)
        fun place(file: ModelFile): ModelFile = if (pinned) file.copy(source = reference.source) else file.copy(path = reference.source + file.path)
        val encoder = place(file(files["encoder"], "adapter.json files.encoder"))
        val decoder = place(file(files["decoder"], "adapter.json files.decoder"))
        if (encoder.bytes + decoder.bytes != reference.sizeBytes) {
            throw ManifestException("adapter.json: its files add up to ${encoder.bytes + decoder.bytes} bytes, model.json says ${reference.sizeBytes}")
        }
        return AdapterManifest(Json.string(json["styleToken"], "adapter.json styleToken"), encoder, decoder)
    }

    fun isPinnedSource(source: String): Boolean =
        PINNED_SOURCE.matches(source) && source.split('/').none { it == "." || it == ".." }

    private fun parse(text: String): JsonElement =
        try {
            KotlinJson.parseToJsonElement(text)
        } catch (error: IllegalArgumentException) {
            throw ManifestException("not valid JSON (${error.message})")
        }

    private fun checkSchema(json: JsonObject) {
        val version = (json["schemaVersion"] as? JsonPrimitive)?.content
        if (version != SUPPORTED_SCHEMA_VERSION.toString()) {
            throw ManifestException("schemaVersion $version is not supported (expected $SUPPORTED_SCHEMA_VERSION)")
        }
    }

    private fun isRelativePath(path: String): Boolean =
        ".." !in path.split('/') && !path.startsWith("/") && '\\' !in path && !URL_SCHEME.containsMatchIn(path)

    private fun file(value: JsonElement?, where: String): ModelFile {
        val json = Json.obj(value, where)
        val path = Json.string(json["path"], "$where.path")
        if (!isRelativePath(path)) throw ManifestException("$where.path must be a relative path inside the model folder")
        val sha256 = Json.string(json["sha256"], "$where.sha256")
        if (!SHA256.matches(sha256)) throw ManifestException("$where.sha256 must be 64 lowercase hex characters")
        val bytes = Json.long(json["bytes"], "$where.bytes", 1)
        val source = Json.optional(json["source"]) ?: return ModelFile(path, bytes, sha256)
        val sourceText = (source as? JsonPrimitive)?.takeIf { it.isString }?.content
        if (sourceText == null || !isPinnedSource(sourceText)) throw ManifestException("$where.source must be a folder at one commit of a Thelivu repository")
        return ModelFile(path, bytes, sha256, sourceText)
    }

    private fun adapterReferences(value: JsonElement?): Map<Style, AdapterReference> {
        val json = Json.obj(value, "adapters")
        val parsed = linkedMapOf<Style, AdapterReference>()
        for ((id, raw) in json) {
            val style = Style.fromId(id) ?: throw ManifestException("adapters.$id: adapters are named after the style they are trained for")
            val reference = Json.obj(raw, "adapters.$id")
            val source = Json.string(reference["source"], "adapters.$id.source")
            if (!isPinnedSource(source) && !(source.endsWith("/") && isRelativePath(source))) {
                throw ManifestException("adapters.$id.source must be a folder inside the model folder, or at one commit of a Thelivu repository")
            }
            parsed[style] = AdapterReference(
                source = source,
                manifest = file(reference["manifest"], "adapters.$id.manifest"),
                sizeBytes = Json.long(reference["sizeBytes"], "adapters.$id.sizeBytes", 1),
            )
        }
        if (parsed.isEmpty()) throw ManifestException("adapters must name at least one adapter")
        return parsed
    }

    private fun prompt(value: JsonElement?): List<PromptSegment> {
        val list = value as? JsonArray
        if (list == null || list.isEmpty()) throw ManifestException("prompt must be a non-empty array")
        val segments = list.mapIndexed { index, raw ->
            val segment = Json.obj(raw, "prompt[$index]")
            when {
                isTrue(segment["input"]) -> PromptSegment.Input
                isTrue(segment["styleToken"]) -> PromptSegment.StyleToken
                (segment["token"] as? JsonPrimitive)?.isString == true -> PromptSegment.Token(Json.string(segment["token"], "prompt[$index].token"))
                else -> throw ManifestException("""prompt[$index] must be {"token"}, {"styleToken": true} or {"input": true}""")
            }
        }
        if (segments.count { it == PromptSegment.Input } != 1) throw ManifestException("""prompt must contain exactly one {"input": true} segment""")
        if (segments.count { it == PromptSegment.StyleToken } != 1) throw ManifestException("""prompt must contain exactly one {"styleToken": true} segment""")
        return segments
    }

    private fun isTrue(value: JsonElement?): Boolean = value is JsonPrimitive && !value.isString && value.content == "true"

    private fun parseBase(json: JsonObject): BaseModel {
        if ((json["ioContract"] as? JsonPrimitive)?.content != SUPPORTED_IO_CONTRACT) {
            throw ManifestException("ioContract ${json["ioContract"]} is not supported (expected $SUPPORTED_IO_CONTRACT)")
        }
        val files = Json.obj(json["files"], "files")
        val architecture = Json.obj(json["architecture"], "architecture")
        val tokens = Json.obj(json["tokens"], "tokens")
        val limits = Json.obj(json["limits"], "limits")
        val generation = Json.optional(json["generation"])?.let { Json.obj(it, "generation") } ?: JsonObject(emptyMap())

        val vocabSize = Json.int(architecture["vocabSize"], "architecture.vocabSize", 2)

        val transforms = Json.optional(json["inputTransforms"])?.let { raw ->
            Json.array(raw, "inputTransforms").map { name ->
                val id = (name as? JsonPrimitive)?.takeIf { it.isString }?.content
                TextTransform.entries.firstOrNull { it.id == id }
                    ?: throw ManifestException("inputTransforms must be a list drawn from ${TextTransform.entries.joinToString { it.id }}")
            }
        } ?: emptyList()

        val reserved = Json.optional(json["reservedInputTokens"])?.let { raw ->
            Json.array(raw, "reservedInputTokens").map { token ->
                (token as? JsonPrimitive)?.takeIf { it.isString }?.content ?: throw ManifestException("reservedInputTokens must be a list of strings")
            }
        } ?: emptyList()

        val repetitionPenalty = Json.optional(generation["repetitionPenalty"])?.let { Json.double(it, "generation.repetitionPenalty") } ?: 1.0
        if (!(repetitionPenalty > 0)) throw ManifestException("generation.repetitionPenalty must be a positive number")

        val exemptIds = Json.optional(generation["noRepeatExemptTokenIds"])?.let { raw ->
            Json.array(raw, "generation.noRepeatExemptTokenIds").map { id ->
                val value = (id as? JsonPrimitive)?.takeUnless { it.isString }?.content?.toIntOrNull()
                if (value == null || value < 0 || value >= vocabSize) {
                    throw ManifestException("generation.noRepeatExemptTokenIds must be a list of token IDs inside the vocabulary")
                }
                value
            }.toSet()
        } ?: emptySet()

        val exemptUrls = Json.optional(generation["noRepeatExemptUrls"])?.let { Json.boolean(it, "generation.noRepeatExemptUrls") } ?: false

        return BaseModel(
            id = Json.string(json["id"], "id"),
            displayName = Json.string(json["displayName"], "displayName"),
            precision = (json["precision"] as? JsonPrimitive)?.takeIf { it.isString }?.content ?: "unknown",
            encoder = file(files["encoder"], "files.encoder"),
            decoder = file(files["decoder"], "files.decoder"),
            tokenizer = file(files["tokenizer"], "files.tokenizer"),
            architecture = Architecture(
                numDecoderLayers = Json.int(architecture["numDecoderLayers"], "architecture.numDecoderLayers", 1),
                numHeads = Json.int(architecture["numHeads"], "architecture.numHeads", 1),
                headDim = Json.int(architecture["headDim"], "architecture.headDim", 1),
                vocabSize = vocabSize,
            ),
            tokens = SpecialTokens(
                decoderStartId = Json.int(tokens["decoderStartId"], "tokens.decoderStartId", 0),
                eosId = Json.int(tokens["eosId"], "tokens.eosId", 0),
                padId = Json.int(tokens["padId"], "tokens.padId", 0),
            ),
            limits = Limits(
                maxInputTokens = Json.int(limits["maxInputTokens"], "limits.maxInputTokens", 1),
                maxNewTokens = Json.int(limits["maxNewTokens"], "limits.maxNewTokens", 1),
            ),
            generation = GenerationSettings(
                repetitionPenalty = repetitionPenalty,
                noRepeatNgramSize = Json.optional(generation["noRepeatNgramSize"])?.let { Json.int(it, "generation.noRepeatNgramSize", 0) } ?: 0,
                noRepeatExemptTokenIds = exemptIds,
                noRepeatExemptUrls = exemptUrls,
            ),
            inputTransforms = transforms,
            reservedInputTokens = reserved,
            prompt = prompt(json["prompt"]),
        )
    }
}
