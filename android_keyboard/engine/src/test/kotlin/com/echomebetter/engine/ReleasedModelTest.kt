package com.echomebetter.engine

import com.echomebetter.engine.manifest.ManifestParser
import com.echomebetter.engine.onnx.CrossKvRunner
import com.echomebetter.engine.onnx.LoraAdapterPair
import com.echomebetter.engine.onnx.OnnxSessions
import com.echomebetter.engine.tokenizer.Tokenizer
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The released model, end to end on this machine's CPU. Opt-in, as it needs the model files (~0.9 GB):
 *
 *     THELIVU_MODEL_DIR=<folder with model.json, encoder.onnx, decoder.onnx, tokenizer.json>
 *     THELIVU_ADAPTER_DIR=<folder with the grammar adapter.json and its two .onnx_adapter files>
 */
class ReleasedModelTest {
    private val modelDir = System.getenv("THELIVU_MODEL_DIR")?.let(::File)
    private val adapterDir = System.getenv("THELIVU_ADAPTER_DIR")?.let(::File)

    @Test
    fun `grammar fixes a sentence and keeps what it must keep`() = runBlocking {
        assumeTrue("THELIVU_MODEL_DIR and THELIVU_ADAPTER_DIR are not set", modelDir != null && adapterDir != null)
        val index = ManifestParser.parseIndex(File(modelDir, "model.json").readText())
        val reference = index.adapters.getValue(Style.GRAMMAR)
        val adapter = ManifestParser.parseAdapterFile(File(adapterDir, "adapter.json").readText(), Style.GRAMMAR, reference, index.base)
        val engine = RewriteEngine(index.base, Tokenizer.fromJson(File(modelDir, index.base.tokenizer.path).readText()))
        val loadStart = System.nanoTime()
        OnnxSessions.open(File(modelDir, index.base.encoder.path), File(modelDir, index.base.decoder.path)).use { sessions ->
            LoraAdapterPair.load(File(adapterDir, adapter.encoder.path.substringAfterLast('/')), File(adapterDir, adapter.decoder.path.substringAfterLast('/'))).use { pair ->
                println("load: ${(System.nanoTime() - loadStart) / 1_000_000} ms")
                val runner = CrossKvRunner(sessions.env, sessions.encoder, sessions.decoder, index.base.architecture).withAdapter(pair)
                for (text in listOf(
                    "hey can u send me the report by tomorow, its kinda urgent",
                    "i has went to the store yesterday and buyed 3 apple for \$4.50",
                    "Please check https://example.com/docs/setup before the meeting at 10:30.",
                )) {
                    val start = System.nanoTime()
                    val result = engine.rewrite(Style.GRAMMAR, adapter.styleToken, runner, text)
                    val ms = (System.nanoTime() - start) / 1_000_000
                    println("[$ms ms, ${result.outputTokens} tokens] $text\n    -> ${result.text}")
                    assertTrue(result.text.isNotBlank())
                }
                val kept = engine.rewrite(Style.GRAMMAR, adapter.styleToken, runner, "Please check https://example.com/docs/setup before the meeting at 10:30.").text
                assertTrue("https://example.com/docs/setup" in kept, kept)
                assertTrue("10:30" in kept, kept)
                assertEquals(kept, engine.rewrite(Style.GRAMMAR, adapter.styleToken, runner, "Please check https://example.com/docs/setup before the meeting at 10:30.").text)
            }
        }
    }
}
