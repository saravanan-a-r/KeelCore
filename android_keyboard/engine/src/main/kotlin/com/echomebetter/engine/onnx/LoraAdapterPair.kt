package com.echomebetter.engine.onnx

import ai.onnxruntime.OrtLoraAdapter
import ai.onnxruntime.OrtSession
import java.io.File

/**
 * A style's adapter, loaded: the LoRA weights for each graph and the run options that make them active. Each
 * session rejects weights it does not have, so the two halves cannot be swapped.
 */
class LoraAdapterPair private constructor(
    private val encoder: OrtLoraAdapter,
    private val decoder: OrtLoraAdapter,
) : AutoCloseable {
    internal val encoderOptions: OrtSession.RunOptions = OrtSession.RunOptions().apply { addActiveLoraAdapter(encoder) }
    internal val decoderOptions: OrtSession.RunOptions = OrtSession.RunOptions().apply { addActiveLoraAdapter(decoder) }

    override fun close() {
        encoderOptions.close()
        decoderOptions.close()
        encoder.close()
        decoder.close()
    }

    companion object {
        fun load(encoderFile: File, decoderFile: File): LoraAdapterPair {
            val encoder = OrtLoraAdapter.create(encoderFile.path)
            var decoder: OrtLoraAdapter? = null
            try {
                decoder = OrtLoraAdapter.create(decoderFile.path)
                return LoraAdapterPair(encoder, decoder)
            } catch (error: Throwable) {
                decoder?.close()
                encoder.close()
                throw error
            }
        }
    }
}
