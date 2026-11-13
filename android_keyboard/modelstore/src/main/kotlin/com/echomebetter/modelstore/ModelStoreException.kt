package com.echomebetter.modelstore

/** Why a model or adapter could not be installed. */
sealed class ModelStoreException(message: String, cause: Throwable? = null) : Exception(message, cause) {
    /** The network failed or the server answered with an error; trying again later may work. */
    class Network(message: String, cause: Throwable? = null) : ModelStoreException(message, cause)

    /** A file arrived with the wrong size or hash. It was deleted; trying again downloads it afresh. */
    class Integrity(message: String) : ModelStoreException(message)

    /** Not enough free space for the files still to download. */
    class InsufficientStorage(val neededBytes: Long, val freeBytes: Long) :
        ModelStoreException("needs $neededBytes bytes, $freeBytes are free")

    /** The published model is a different base than the installed one: the base has to be installed again first. */
    class BaseChanged : ModelStoreException("the published model is a newer base than the installed one")

    /** The published model has no adapter for this style. */
    class NoSuchAdapter(style: String) : ModelStoreException("the published model has no $style adapter")

    /** The adapter needs the base model, which is not installed. */
    class BaseMissing : ModelStoreException("the base model is not installed")
}
