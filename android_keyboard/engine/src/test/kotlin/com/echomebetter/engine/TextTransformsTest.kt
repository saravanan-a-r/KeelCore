package com.echomebetter.engine

import com.echomebetter.engine.manifest.TextTransform
import com.echomebetter.engine.prompt.TextTransforms
import org.junit.Test
import kotlin.test.assertEquals

class TextTransformsTest {
    private val transforms = listOf(TextTransform.ESCAPE_SPIECE_MARKERS)

    @Test
    fun `escaping is lossless for any text, the escape character included`() {
        for (text in listOf("plain text", "a▁b", "", "▁", " alone", "")) {
            assertEquals(text, TextTransforms.undo(TextTransforms.apply(text, transforms), transforms))
        }
        assertEquals("ab", TextTransforms.apply("a▁b", transforms))
        assertEquals("", TextTransforms.apply("", transforms))
    }
}
