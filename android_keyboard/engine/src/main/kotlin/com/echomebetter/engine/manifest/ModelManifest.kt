package com.echomebetter.engine.manifest

import com.echomebetter.engine.Style

/**
 * `model.json`: everything model-specific, as data. The same schema the extension reads
 * (chrome_extension/src/engine/manifest.ts): the base's files and settings in full, and one reference per style to
 * the folder of its adapter, which holds an `adapter.json` naming the adapter's files.
 */
class ManifestException(message: String) : Exception("model.json: $message")

/** A file of the model, by path relative to the model's folder or to [source]. */
data class ModelFile(
    val path: String,
    val bytes: Long,
    val sha256: String,
    /** The folder [path] is relative to when it is not the model's own: one commit of a Thelivu repository. */
    val source: String? = null,
)

/** A model.json's entry for one adapter: where it is, and the hash that vouches for its adapter.json. */
data class AdapterReference(
    /** A folder: relative to the model's, or pinned to one commit of a Thelivu repository. */
    val source: String,
    /** The adapter.json, with its path relative to [source]. */
    val manifest: ModelFile,
    /** The size of the adapter's files, for showing before anything is fetched. */
    val sizeBytes: Long,
)

/** An adapter.json, resolved against the model that refers to it. */
data class AdapterManifest(
    /** The style marker the adapter was trained with. */
    val styleToken: String,
    val encoder: ModelFile,
    val decoder: ModelFile,
) {
    val files: List<ModelFile> get() = listOf(encoder, decoder)
}

sealed interface PromptSegment {
    /** One vocabulary piece by name (a frame marker, `</s>`). */
    data class Token(val token: String) : PromptSegment

    /** The style token of the adapter that runs. */
    data object StyleToken : PromptSegment

    /** The user's text. */
    data object Input : PromptSegment
}

data class Architecture(val numDecoderLayers: Int, val numHeads: Int, val headDim: Int, val vocabSize: Int)

data class SpecialTokens(val decoderStartId: Int, val eosId: Int, val padId: Int)

data class Limits(val maxInputTokens: Int, val maxNewTokens: Int)

data class GenerationSettings(
    val repetitionPenalty: Double = 1.0,
    val noRepeatNgramSize: Int = 0,
    /** Token IDs the n-gram ban never applies to (the digits). */
    val noRepeatExemptTokenIds: Set<Int> = emptySet(),
    /** Lift the n-gram ban while a URL is being written. */
    val noRepeatExemptUrls: Boolean = false,
)

enum class TextTransform(val id: String) {
    ESCAPE_SPIECE_MARKERS("escape-spiece-markers"),
}

/** Everything model.json says about the base model. */
data class BaseModel(
    val id: String,
    val displayName: String,
    val precision: String,
    val encoder: ModelFile,
    val decoder: ModelFile,
    val tokenizer: ModelFile,
    val architecture: Architecture,
    val tokens: SpecialTokens,
    val limits: Limits,
    val generation: GenerationSettings,
    val inputTransforms: List<TextTransform>,
    val reservedInputTokens: List<String>,
    val prompt: List<PromptSegment>,
) {
    /** The base's files, smallest first. */
    val files: List<ModelFile> get() = listOf(tokenizer, encoder, decoder)

    val sizeBytes: Long get() = files.sumOf { it.bytes }

    /** Identifies the base weights: adapters are trained against one base and name it in their adapter.json. */
    val fingerprint: String get() = files.joinToString(":") { it.sha256 }
}

/** A model.json as published: the base in full, the adapters by reference. */
data class ModelIndex(
    val base: BaseModel,
    val adapters: Map<Style, AdapterReference>,
)

/** A base with the adapters that are on hand, ready to run. */
data class ModelManifest(
    val base: BaseModel,
    val adapters: Map<Style, AdapterManifest>,
)
