package com.echomebetter.rewrite.access

import com.echomebetter.engine.Style

/**
 * Which styles the user may use. Grammar is free; the other four are one purchase. Until Play Billing is wired in,
 * [unlockAll] (on in development builds) is the only way to unlock them.
 */
class StyleAccess(private val unlockAll: Boolean) {
    fun isFree(style: Style): Boolean = style in FREE

    fun isUnlocked(style: Style): Boolean = unlockAll || isFree(style)

    companion object {
        val FREE: Set<Style> = setOf(Style.GRAMMAR)
    }
}
