import type { Compiler } from 'webpack';
import { LICENSE_DATE, LICENSE_NAME, LICENSE_VERSION } from '../src/shared/license';

/**
 * The notice written at the top of every JavaScript file the build emits.
 * It carries the essentials of the EchoMeBetter License (LICENSE at the
 * project root); that file controls if anything here seems different. When
 * the license changes, its version and date change in src/shared/license.ts.
 */
export const LICENSE_BANNER = `EchoMeBetter
Copyright (c) 2026 Saravanan A R. All rights reserved.
https://github.com/saravanan-a-r/EchoMeBetter

Licensed under the ${LICENSE_NAME}, Version ${LICENSE_VERSION} (Personal and Research
Use), dated ${LICENSE_DATE}. This is not open source software. The full
terms are in the LICENSE file at the address above, also included in this
extension as LICENSE.txt, and that file controls.

Allowed: use by one individual, free of charge, only through the official
EchoMeBetter browser extension, for their own writing (including writing for
their own job or studies) and for noncommercial research, which must cite
the project.

Not allowed: copying, extracting, modifying, reverse engineering,
redistributing, sharing, hosting or selling this code or the EchoMeBetter
model; using them anywhere other than the official extension; using them, or
their output, to build or train other models; using them through scripts or
bots; or any commercial use, direct or indirect.

Commercial use, use by or for an organization, use outside the official
extension and purchase of the model require a separate written agreement:
sararavi.ar@gmail.com

Third-party components keep their own licenses (see the *.LICENSE.txt files
next to the scripts that include them).

THE SOFTWARE AND THE MODEL ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.`;


const PLUGIN = 'EchoMeBetterLicenseBanner';

/** The notice as one comment block. */
function bannerComment(): string {
  return `/*!\n${LICENSE_BANNER.split('\n')
    .map((line) => (line ? ` * ${line}` : ' *'))
    .join('\n')}\n */\n`;
}

/**
 * Webpack plugin: puts the notice at the top of every JavaScript file the
 * build emits. It runs after the minifier, which would otherwise move a
 * comment like this out into a .LICENSE.txt file and leave only a pointer.
 * Every emitted .js asset gets it, whatever chunk it belongs to; files the
 * build only copies (the onnxruntime vendor files) are never touched.
 */
export class LicenseBannerPlugin {
  apply(compiler: Compiler): void {
    const { ConcatSource } = compiler.webpack.sources;
    const comment = bannerComment();

    compiler.hooks.thisCompilation.tap(PLUGIN, (compilation) => {
      compilation.hooks.processAssets.tap({ name: PLUGIN, stage: compiler.webpack.Compilation.PROCESS_ASSETS_STAGE_REPORT }, (assets) => {
        for (const file of Object.keys(assets)) {
          if (!file.endsWith('.js')) continue;
          compilation.updateAsset(file, (old) => (String(old.source()).startsWith(comment) ? old : new ConcatSource(comment, old)));
        }
      });
    });
  }
}
