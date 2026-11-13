package com.echomebetter.modelstore

import com.echomebetter.engine.Style
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ModelInstallerTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val fixture = File(requireNotNull(javaClass.getResource("/tiny-echo")).toURI())
    private val server = FixtureServer(fixture)
    private val store by lazy { ModelStore(temp.newFolder("models")) }
    private val installer by lazy { ModelInstaller(store, server.url, Downloader("test", attempts = 3, retryDelayMs = 1)) }

    @After
    fun stop() = server.close()

    private fun installBase() = runBlocking { installer.installBase {} }

    @Test
    fun `installs the base, then adapters on top, with every file in place and progress to the total`() = runBlocking {
        assertNull(store.installed())
        val progress = ArrayList<InstallProgress>()
        val index = installer.installBase { progress.add(it) }
        assertEquals(index.base.sizeBytes, progress.last().doneBytes)
        assertEquals(progress.last().totalBytes, progress.last().doneBytes)

        val installed = assertNotNull(store.installed())
        assertEquals(index.base.fingerprint, installed.index.base.fingerprint)
        assertTrue(installed.manifest.adapters.isEmpty())
        assertEquals(index.base.encoder.bytes, installed.encoder.length())

        installer.installAdapter(Style.GRAMMAR) {}
        val withGrammar = assertNotNull(store.installed())
        assertEquals(setOf(Style.GRAMMAR), withGrammar.manifest.adapters.keys)
        val (encoder, decoder) = assertNotNull(withGrammar.adapterFiles(Style.GRAMMAR))
        assertEquals(File(fixture, "adapters/grammar/encoder.onnx_adapter").readBytes().toList(), encoder.readBytes().toList())
        assertTrue(decoder.isFile)
        assertFalse(File(store.root, "staging/adapters/grammar").exists())
    }

    @Test
    fun `installing the base again with the same base downloads nothing`() {
        installBase()
        server.requests.clear()
        installBase()
        assertEquals(listOf("model.json"), server.requests.map { it.first })
    }

    @Test
    fun `an interrupted download resumes from the bytes it has`() {
        server.cutAfter["base/encoder.onnx"] = 10_000
        installBase()
        val encoderRequests = server.requests.filter { it.first == "base/encoder.onnx" }
        assertEquals(listOf(null, "bytes=10000-"), encoderRequests.map { it.second })
        assertNotNull(store.installed())
    }

    @Test
    fun `a file with the wrong hash is deleted, and nothing counts as installed`() {
        server.corrupt.add("base/decoder.onnx")
        assertFailsWith<ModelStoreException.Integrity> { installBase() }
        assertNull(store.installed())
        assertFalse(File(store.root, "staging/base/decoder.onnx").exists())
        assertFalse(File(store.root, "staging/base/decoder.onnx.part").exists())
        // Fixed on the server: the next try succeeds, and the files already verified are not fetched again.
        server.corrupt.clear()
        server.requests.clear()
        installBase()
        assertNotNull(store.installed())
        assertFalse(server.requests.any { it.first == "base/encoder.onnx" })
    }

    @Test
    fun `an adapter json that does not match model json is refused`() {
        installBase()
        server.overrides["adapters/grammar/adapter.json"] = File(fixture, "adapters/grammar/adapter.json").readText().replace("grammar", "grammar ").toByteArray()
        assertFailsWith<ModelStoreException.Integrity> { runBlocking { installer.installAdapter(Style.GRAMMAR) {} } }
        assertTrue(assertNotNull(store.installed()).manifest.adapters.isEmpty())
    }

    @Test
    fun `an adapter needs the base, and a base that changed upstream`() {
        assertFailsWith<ModelStoreException.BaseMissing> { runBlocking { installer.installAdapter(Style.GRAMMAR) {} } }
        installBase()
        val modelJson = File(fixture, "model.json").readText()
        val tokenizerHash = Regex(""""tokenizer":\s*\{[^}]*"sha256":\s*"([0-9a-f]{64})"""").find(modelJson)!!.groupValues[1]
        server.overrides["model.json"] = modelJson.replace(tokenizerHash, "0".repeat(64)).toByteArray()
        assertFailsWith<ModelStoreException.BaseChanged> { runBlocking { installer.installAdapter(Style.GRAMMAR) {} } }
    }

    @Test
    fun `a style the model does not publish is reported as such`() {
        installBase()
        assertFailsWith<ModelStoreException.NoSuchAdapter> { runBlocking { installer.installAdapter(Style.CONCISE) {} } }
    }

    @Test
    fun `removing an adapter keeps the base, removing all leaves nothing`() = runBlocking {
        installer.installBase {}
        installer.installAdapter(Style.GRAMMAR) {}
        installer.installAdapter(Style.PROFESSIONAL) {}
        store.removeAdapter(Style.GRAMMAR)
        assertEquals(setOf(Style.PROFESSIONAL), assertNotNull(store.installed()).manifest.adapters.keys)
        store.removeAll()
        assertNull(store.installed())
        assertTrue(store.root.listFiles().isNullOrEmpty())
    }

    @Test
    fun `a damaged install does not count as installed`() = runBlocking {
        installer.installBase {}
        installer.installAdapter(Style.GRAMMAR) {}
        val installed = assertNotNull(store.installed())
        val (encoder, _) = assertNotNull(installed.adapterFiles(Style.GRAMMAR))
        encoder.writeBytes(ByteArray(3))
        assertTrue(assertNotNull(store.installed()).manifest.adapters.isEmpty())
        installed.decoder.delete()
        assertNull(store.installed())
    }
}
