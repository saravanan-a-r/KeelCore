package com.echomebetter.rewrite.model

import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequest
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.echomebetter.engine.Style
import com.echomebetter.engine.manifest.ModelIndex
import com.echomebetter.modelstore.InstallProgress
import com.echomebetter.modelstore.ModelInstaller
import com.echomebetter.modelstore.ModelStore
import com.echomebetter.modelstore.ModelStoreException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

/**
 * The model on the phone and the downloads that change it.
 *
 * Downloads run as WorkManager jobs ([InstallWorker]) in one unique chain, so they run one after another and survive
 * the app being closed. Every change to the files (an install, a removal) holds one lock, so a removal never races
 * a download. [installed] is re-read after each change; the keyboard and the setup screens both watch it.
 */
class ModelRepository(
    private val store: ModelStore,
    private val installer: ModelInstaller,
    workManager: Lazy<WorkManager>,
    private val scope: CoroutineScope,
) {
    private val workManager by workManager
    private val lock = Mutex()

    private val _installed = MutableStateFlow<InstalledState>(InstalledState.Checking)
    val installed: StateFlow<InstalledState> = _installed.asStateFlow()

    private val _catalog = MutableStateFlow<ModelIndex?>(null)

    /** The published model.json, once fetched: what each download costs. Null until then or when offline. */
    val catalog: StateFlow<ModelIndex?> = _catalog.asStateFlow()

    init {
        scope.launch { refresh() }
    }

    /** Re-reads what is installed. */
    suspend fun refresh() {
        val model = withContext(Dispatchers.IO) { store.installed() }
        _installed.value = if (model == null) InstalledState.None else InstalledState.Ready(model)
    }

    /** Fetches the published model.json for its sizes; a failure leaves [catalog] as it was. */
    suspend fun refreshCatalog() {
        try {
            _catalog.value = installer.fetchIndex()
        } catch (_: ModelStoreException) {
        } catch (_: com.echomebetter.engine.manifest.ManifestException) {
        }
    }

    /** Free space for the model files. */
    suspend fun freeBytes(): Long = withContext(Dispatchers.IO) { store.freeBytes() }

    /** Queues [targets] in order, behind any download already queued. */
    fun install(targets: List<InstallTarget>, allowMetered: Boolean) {
        if (targets.isEmpty()) return
        val requests = targets.map { request(it, allowMetered) }
        workManager.beginUniqueWork(UNIQUE_WORK, ExistingWorkPolicy.APPEND_OR_REPLACE, requests).enqueue()
    }

    /** Stops every queued and running download; partial files stay, so a later download resumes. */
    fun cancelDownloads() {
        workManager.cancelUniqueWork(UNIQUE_WORK)
    }

    /** Each target's download, while queued, running or failed. */
    val downloads: Flow<Map<InstallTarget, DownloadState>>
        get() = workManager.getWorkInfosForUniqueWorkFlow(UNIQUE_WORK).map { infos ->
            infos.mapNotNull { info -> targetOf(info)?.let { target -> stateOf(info)?.let { target to it } } }.toMap()
        }

    suspend fun removeAdapter(style: Style) {
        workManager.cancelAllWorkByTag(tagOf(InstallTarget.Adapter(style)))
        lock.withLock { withContext(Dispatchers.IO) { store.removeAdapter(style) } }
        refresh()
    }

    suspend fun removeAll() {
        cancelDownloads()
        lock.withLock { withContext(Dispatchers.IO) { store.removeAll() } }
        refresh()
    }

    /** Runs one install; called by [InstallWorker]. */
    internal suspend fun runInstall(target: InstallTarget, onProgress: (InstallProgress) -> Unit) {
        try {
            lock.withLock {
                when (target) {
                    InstallTarget.Base -> installer.installBase(onProgress)
                    is InstallTarget.Adapter -> installer.installAdapter(target.style, onProgress)
                }
            }
        } finally {
            refresh()
        }
    }

    private fun request(target: InstallTarget, allowMetered: Boolean): OneTimeWorkRequest =
        OneTimeWorkRequestBuilder<InstallWorker>()
            .setInputData(workDataOf(InstallWorker.KEY_TARGET to target.key))
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(if (allowMetered) NetworkType.CONNECTED else NetworkType.UNMETERED)
                    .setRequiresStorageNotLow(true)
                    .build(),
            )
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .addTag(TAG)
            .addTag(tagOf(target))
            .build()

    private fun targetOf(info: WorkInfo): InstallTarget? =
        info.tags.firstOrNull { it.startsWith(TARGET_TAG_PREFIX) }?.let { InstallTarget.parse(it.removePrefix(TARGET_TAG_PREFIX)) }

    private fun stateOf(info: WorkInfo): DownloadState? = when (info.state) {
        WorkInfo.State.ENQUEUED, WorkInfo.State.BLOCKED ->
            DownloadState.Queued(waitingForWifi = info.constraints.requiredNetworkType == NetworkType.UNMETERED)
        WorkInfo.State.RUNNING -> DownloadState.Running(
            doneBytes = info.progress.getLong(InstallWorker.KEY_DONE, 0),
            totalBytes = info.progress.getLong(InstallWorker.KEY_TOTAL, 0),
        )
        WorkInfo.State.FAILED -> DownloadState.Failed(
            reason = info.outputData.getString(InstallWorker.KEY_REASON)?.let { name -> FailureReason.entries.firstOrNull { it.name == name } } ?: FailureReason.UNKNOWN,
            neededBytes = info.outputData.getLong(InstallWorker.KEY_NEEDED, 0),
        )
        WorkInfo.State.SUCCEEDED, WorkInfo.State.CANCELLED -> null
    }

    private companion object {
        const val UNIQUE_WORK = "model-install"
        const val TAG = "model-install"
        const val TARGET_TAG_PREFIX = "target="

        fun tagOf(target: InstallTarget) = TARGET_TAG_PREFIX + target.key
    }
}
