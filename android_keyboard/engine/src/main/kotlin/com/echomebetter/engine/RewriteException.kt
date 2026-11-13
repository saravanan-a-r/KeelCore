package com.echomebetter.engine

/** A rewrite that cannot be done, for a reason the user can act on. Cancellation is a [kotlinx.coroutines.CancellationException], not one of these. */
sealed class RewriteException(message: String) : Exception(message) {
    /** The text is longer than the model takes, even cut into pieces. */
    class InputTooLong(val actual: Int, val limit: Int) : RewriteException("input is $actual tokens, the limit is $limit")

    /** The model kept writing past `limits.maxNewTokens`, also after the text was halved. */
    class OutputTooLong(val limit: Int) : RewriteException("the rewrite ran past $limit tokens")

    /** The model wrote nothing. */
    class EmptyResult : RewriteException("the rewrite is empty")

    /** The text spells one of the model's control markers, which would break the prompt frame. */
    class ReservedMarkup : RewriteException("the text contains the model's reserved markup")
}
