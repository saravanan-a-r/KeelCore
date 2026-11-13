package com.echomebetter.rewrite

import androidx.annotation.StringRes
import com.echomebetter.engine.Style

/** A style's name and one-line description, as the keyboard and the setup screens show them. */
@get:StringRes
val Style.label: Int
    get() = when (this) {
        Style.GRAMMAR -> R.string.style_grammar
        Style.PROFESSIONAL -> R.string.style_professional
        Style.FRIENDLY -> R.string.style_friendly
        Style.CONCISE -> R.string.style_concise
        Style.ELABORATE -> R.string.style_elaborate
    }

@get:StringRes
val Style.hint: Int
    get() = when (this) {
        Style.GRAMMAR -> R.string.style_grammar_hint
        Style.PROFESSIONAL -> R.string.style_professional_hint
        Style.FRIENDLY -> R.string.style_friendly_hint
        Style.CONCISE -> R.string.style_concise_hint
        Style.ELABORATE -> R.string.style_elaborate_hint
    }
