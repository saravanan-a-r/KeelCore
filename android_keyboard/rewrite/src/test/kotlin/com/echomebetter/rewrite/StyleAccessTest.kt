package com.echomebetter.rewrite

import com.echomebetter.engine.Style
import com.echomebetter.rewrite.access.StyleAccess
import org.junit.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class StyleAccessTest {
    @Test
    fun `grammar is free, the rest are unlocked only when every style is`() {
        val release = StyleAccess(unlockAll = false)
        assertTrue(release.isUnlocked(Style.GRAMMAR))
        assertTrue(release.isFree(Style.GRAMMAR))
        assertFalse(release.isUnlocked(Style.PROFESSIONAL))
        assertTrue(Style.entries.all(StyleAccess(unlockAll = true)::isUnlocked))
    }
}
