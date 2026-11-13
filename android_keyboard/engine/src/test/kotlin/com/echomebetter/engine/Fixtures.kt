package com.echomebetter.engine

import com.echomebetter.engine.manifest.AdapterManifest
import com.echomebetter.engine.manifest.ManifestParser
import com.echomebetter.engine.manifest.ModelIndex
import com.echomebetter.engine.onnx.CrossKvRunner
import com.echomebetter.engine.onnx.LoraAdapterPair
import com.echomebetter.engine.onnx.OnnxSessions
import com.echomebetter.engine.tokenizer.Tokenizer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/** The extension's test models (chrome_extension/src/test/fixtures), copied into this module's test resources. */
object Fixtures {
    fun dir(name: String): File = File(requireNotNull(Fixtures::class.java.getResource("/$name")) { "missing fixture $name" }.toURI())

    fun text(name: String): String = File(dir(name.substringBeforeLast('/')), name.substringAfterLast('/')).readText()

    /** A whole fixture model: index, tokenizer, resolved adapters and open sessions. Close it after use. */
    class Model(val folder: File) : AutoCloseable {
        val index: ModelIndex = ManifestParser.parseIndex(File(folder, "model.json").readText())
        val tokenizer: Tokenizer = Tokenizer.fromJson(File(folder, index.base.tokenizer.path).readText())
        val adapters: Map<Style, AdapterManifest> = index.adapters.mapValues { (style, reference) ->
            ManifestParser.parseAdapterFile(File(folder, reference.source + reference.manifest.path).readText(), style, reference, index.base)
        }
        val sessions = OnnxSessions.open(File(folder, index.base.encoder.path), File(folder, index.base.decoder.path))
        val baseRunner = CrossKvRunner(sessions.env, sessions.encoder, sessions.decoder, index.base.architecture)
        val engine = RewriteEngine(index.base, tokenizer)

        /** The adapter a style runs with: its own, else the model's fallback (professional in the fixtures). */
        fun adapterStyle(style: Style): Style = if (style in adapters) style else Style.PROFESSIONAL

        fun loadAdapter(style: Style): LoraAdapterPair {
            val adapter = adapters.getValue(style)
            return LoraAdapterPair.load(File(folder, adapter.encoder.path), File(folder, adapter.decoder.path))
        }

        override fun close() = sessions.close()
    }

    class GenerationCase(val style: String, val input: String, val ids: List<Int>, val maxNewTokens: Int, val outputIds: List<Int>, val outputText: String, val baseOutputIds: List<Int>)

    fun parity(name: String): List<GenerationCase> {
        val json = Json.parseToJsonElement(text("$name/parity.json")).jsonObject
        fun ints(obj: JsonObject, key: String) = obj.getValue(key).jsonArray.map { it.jsonPrimitive.int }
        return json.getValue("generation").jsonArray.map {
            val case = it.jsonObject
            GenerationCase(
                style = case.getValue("style").jsonPrimitive.content,
                input = case.getValue("input").jsonPrimitive.content,
                ids = ints(case, "ids"),
                maxNewTokens = case.getValue("maxNewTokens").jsonPrimitive.int,
                outputIds = ints(case, "outputIds"),
                outputText = case.getValue("outputText").jsonPrimitive.content,
                baseOutputIds = ints(case, "baseOutputIds"),
            )
        }
    }
}
