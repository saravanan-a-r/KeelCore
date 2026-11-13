package com.echomebetter.rewrite

import android.content.Context
import androidx.work.WorkManager
import com.echomebetter.rewrite.access.StyleAccess
import com.echomebetter.rewrite.engine.EngineHost
import com.echomebetter.rewrite.model.ModelRepository
import com.echomebetter.rewrite.settings.RewriteSettings
import com.echomebetter.modelstore.Downloader
import com.echomebetter.modelstore.ModelInstaller
import com.echomebetter.modelstore.ModelStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import java.io.File

/** What the app decides about rewriting, from its build configuration. */
data class RewriteConfig(
    /** The folder the published model.json is in. */
    val modelUrl: String,
    /** Sent with every download request. */
    val userAgent: String,
    /** Development builds unlock every style; release builds wait for the purchase. */
    val unlockAllStyles: Boolean,
    /** Devices with less memory are told the model will not run on them. */
    val minDeviceMemoryBytes: Long,
)

/** The application, as the place the rewrite feature's single [RewriteContainer] lives. */
interface RewriteHost {
    val rewrite: RewriteContainer
}

/** The rewrite feature's shared objects, built once per process by the application. */
class RewriteContainer(context: Context, val config: RewriteConfig) {
    private val appContext = context.applicationContext

    /** Work that outlives a screen or the keyboard's window: refreshing state, unloading the model when idle. */
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val settings = RewriteSettings(appContext)

    val access = StyleAccess(unlockAll = config.unlockAllStyles)

    // Out of backups (noBackupFilesDir), and private to the app.
    private val store = ModelStore(File(appContext.noBackupFilesDir, "thelivu"))

    private val installer = ModelInstaller(store, config.modelUrl, Downloader(userAgent = config.userAgent))

    val models = ModelRepository(store, installer, lazy { WorkManager.getInstance(appContext) }, scope)

    val engine = EngineHost(models, scope)
}

/** The rewrite feature of the running app. */
val Context.rewrite: RewriteContainer
    get() = (applicationContext as? RewriteHost)?.rewrite ?: error("the application does not host the rewrite feature")
