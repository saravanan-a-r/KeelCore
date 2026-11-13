package com.echomebetter.engine

import com.echomebetter.engine.generation.LogitsProcessors
import com.echomebetter.engine.generation.NgramAllowance
import com.echomebetter.engine.generation.UrlTracker
import com.echomebetter.engine.manifest.GenerationSettings
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** Ported from the extension's logitsProcessors.test.ts. */
class LogitsProcessorsTest {
    @Test
    fun `repetition penalty divides positive and multiplies negative scores of seen tokens`() {
        val scores = floatArrayOf(2f, -2f, 4f)
        LogitsProcessors.applyRepetitionPenalty(scores, listOf(0, 1, 0), 2.0)
        assertEquals(listOf(1f, -4f, 4f), scores.toList())
    }

    @Test
    fun `the n-gram ban covers the token that would complete a seen n-gram`() {
        assertEquals(listOf(3), LogitsProcessors.bannedNgramTokens(listOf(1, 2, 3, 1, 2), 3))
        // Size 1 bans every token already in the sequence.
        assertEquals(setOf(5, 6), LogitsProcessors.bannedNgramTokens(listOf(5, 6, 5), 1).toSet())
        // Nothing is banned until the sequence is long enough.
        assertEquals(emptyList(), LogitsProcessors.bannedNgramTokens(listOf(1), 3))
    }

    @Test
    fun `an exempt token is never banned, so a digit run can repeat its window`() {
        val allowance = NgramAllowance(exempt = setOf(0))
        assertEquals(emptyList(), LogitsProcessors.bannedNgramTokens(listOf(5, 0, 0, 0, 0), 3, allowance))
        assertEquals(listOf(0), LogitsProcessors.bannedNgramTokens(listOf(5, 0, 0, 0, 0), 3))
    }

    @Test
    fun `banned tokens become negative infinity`() {
        val scores = floatArrayOf(0f, 1f, 2f, 3f)
        LogitsProcessors.process(scores, listOf(1, 2, 3, 1, 2), GenerationSettings(noRepeatNgramSize = 3), null)
        assertEquals(Float.NEGATIVE_INFINITY, scores[3])
        assertEquals(2, LogitsProcessors.argmax(scores))
    }

    // 3 "▁https"  4 "://"  5 "ex"  6 ".com"  7 "/a"  8 "▁and"  9 "▁the"  10 "▁www"
    private val pieces = mapOf(3 to "▁https", 4 to "://", 5 to "ex", 6 to ".com", 7 to "/a", 8 to "▁and", 9 to "▁the", 10 to "▁www")
    private val urlSettings = GenerationSettings(noRepeatNgramSize = 3, noRepeatExemptUrls = true)

    @Test
    fun `pieceText turns the space mark into a space and byte pieces into their ASCII character`() {
        assertEquals(" and", LogitsProcessors.pieceText("▁and"))
        assertEquals("/", LogitsProcessors.pieceText("<0x2F>"))
        assertEquals("�", LogitsProcessors.pieceText("<0xC3>"))
        assertEquals("://", LogitsProcessors.pieceText("://"))
    }

    @Test
    fun `the URL tracker reads words as the extension does`() {
        val cases = listOf(
            listOf(3) to true, listOf(3, 4) to true, listOf(3, 4, 5, 6, 7) to true, listOf(3, 4, 5, 8) to false,
            listOf(3, 4, 5, 8, 3, 4) to true, listOf(10) to true, listOf(8) to false, listOf(10, 6) to true,
            listOf(5, 6, 7) to true, listOf(5, 6) to false, listOf(9, 5) to false,
        )
        for ((tokens, inside) in cases) {
            val tracker = UrlTracker { LogitsProcessors.pieceText(pieces[it] ?: "") }
            tokens.forEach(tracker::observe)
            assertEquals(inside, tracker.inside, "$tokens")
        }
    }

    @Test
    fun `the ban is lifted inside a URL and applies again after it`() {
        val allowance = NgramAllowance.of(urlSettings, pieces::get, listOf(3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7))
        listOf(3, 4, 5, 6, 7, 8, 3, 4, 5).forEach { allowance.urls!!.observe(it) }
        assertEquals(emptyList(), LogitsProcessors.bannedNgramTokens(listOf(0, 3, 4, 5, 6, 7, 8, 3, 4, 5), 3, allowance))
        assertEquals(listOf(6), LogitsProcessors.bannedNgramTokens(listOf(0, 3, 4, 5, 6, 7, 8, 3, 4, 5), 3))
        val text = NgramAllowance.of(urlSettings, pieces::get, listOf(8, 9, 8, 9, 8, 9))
        listOf(8, 9, 8, 9).forEach { text.urls!!.observe(it) }
        assertEquals(listOf(9), LogitsProcessors.bannedNgramTokens(listOf(0, 8, 9, 8, 9, 8), 3, text))
    }

    @Test
    fun `a piece repeats inside a URL only as often as the input has it`() {
        val once = NgramAllowance.of(urlSettings, pieces::get, listOf(3, 4, 5, 6, 7))
        listOf(3, 4, 5).forEach { once.urls!!.observe(it) }
        assertEquals(emptyList(), LogitsProcessors.bannedNgramTokens(listOf(0, 3, 4, 5), 3, once))
        listOf(6, 7, 8, 3, 4, 5).forEach { once.urls!!.observe(it) }
        assertEquals(listOf(6), LogitsProcessors.bannedNgramTokens(listOf(0, 3, 4, 5, 6, 7, 3, 4, 5), 3, once))
        val none = NgramAllowance.of(urlSettings, pieces::get, listOf(9))
        listOf(3, 4, 5).forEach { none.urls!!.observe(it) }
        assertEquals(listOf(6), LogitsProcessors.bannedNgramTokens(listOf(0, 3, 4, 5, 6, 7, 3, 4, 5), 3, none))
    }

    @Test
    fun `without the setting there is no URL tracking`() {
        assertNull(NgramAllowance.of(urlSettings.copy(noRepeatExemptUrls = false), pieces::get, emptyList()).urls)
    }

    @Test
    fun `argmax returns the first index on ties`() {
        assertEquals(1, LogitsProcessors.argmax(floatArrayOf(1f, 3f, 3f, 2f)))
    }
}
