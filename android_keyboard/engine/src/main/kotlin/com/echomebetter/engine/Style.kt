package com.echomebetter.engine

/**
 * The rewrite styles, one LoRA adapter each. [id] is the name the model's manifest uses
 * (`adapters.<id>` in model.json); the enum order is the order the keyboard shows them in.
 */
enum class Style(val id: String) {
    GRAMMAR("grammar"),
    PROFESSIONAL("professional"),
    FRIENDLY("friendly"),
    CONCISE("concise"),
    ELABORATE("elaborate");

    /**
     * The most output tokens this style writes per input token, so a text goes through one pass only when its
     * rewrite fits `limits.maxNewTokens`; a longer text is cut into pieces first. The values and their evidence
     * are the extension's (chrome_extension/src/engine/rewriteEngine.ts `MAX_EXPANSION`); keep the two equal.
     */
    val maxExpansion: Double
        get() = when (this) {
            PROFESSIONAL -> 1.6
            CONCISE -> 1.15
            ELABORATE -> 2.2
            FRIENDLY -> 1.7
            GRAMMAR -> 1.3
        }

    companion object {
        fun fromId(id: String): Style? = entries.firstOrNull { it.id == id }
    }
}
