// Post-process notebook output cells in the rendered HTML.
//
// nbconvert turns each notebook stdout/result flush into a separate
// ".. parsed-literal::" directive, so a single code cell that prints
// incrementally (e.g. an optimizer logging every iteration) becomes dozens of
// consecutive output boxes. Sphinx renders each one as
//   <div class="highlight-default notranslate"><div class="highlight"><pre>...</pre></div></div>
// This script:
//   1. Merges a run of consecutive output blocks into a single box (one border,
//      one "Output" label -- styling comes from custom.css).
//   2. Makes any tall output (merged group or a single long block) collapsible
//      with a "Show more" / "Show less" toggle.
//
// It runs entirely in the browser on the static HTML produced by sphinx-build,
// so it applies automatically on every build without modifying the .rst files.

(function () {
    "use strict";

    // Height (px) above which an output block is collapsed. ~15em / ~10 lines.
    var COLLAPSE_PX = 240;

    // parsed-literal outputs render as <div class="highlight-default">. Code
    // *inputs* use "highlight-ipython3" and are intentionally left alone.
    var OUTPUT_SELECTOR = "div.highlight-default";

    // Returns the next element sibling, skipping over whitespace-only text
    // nodes (the newlines docutils emits between block-level elements).
    function nextBlockSibling(node) {
        var sib = node.nextSibling;
        while (sib && sib.nodeType === Node.TEXT_NODE && !/\S/.test(sib.nodeValue)) {
            sib = sib.nextSibling;
        }
        return sib && sib.nodeType === Node.ELEMENT_NODE ? sib : null;
    }

    // The <pre> that actually holds the text inside an output block.
    function preOf(block) {
        return block.querySelector("pre");
    }


    // Strip raw ANSI colour escape sequences (e.g. "\x1b[39m") that some
    // libraries' coloured loggers emit. nbconvert keeps them verbatim in the
    // .rst, so without this they render as literal "[39m" noise in the output.
    var ANSI_RE = /\x1b\[[0-9;]*m/g;
    function stripAnsiCodes(root) {
        var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
        var node;
        while ((node = walker.nextNode())) {
            if (node.nodeValue.indexOf("\x1b") !== -1) {
                node.nodeValue = node.nodeValue.replace(ANSI_RE, "");
            }
        }
    }


    // Merge each maximal run of adjacent output blocks into the first block of
    // the run, concatenating the inner <pre> text with a blank line between
    // the original blocks.
    function mergeConsecutiveOutputs(root) {
        var blocks = root.querySelectorAll(OUTPUT_SELECTOR);
        blocks.forEach(function (block) {
            // Skip blocks already absorbed into an earlier run.
            if (block.dataset.pqMerged === "into") {
                return;
            }
            var targetPre = preOf(block);
            if (!targetPre) {
                return;
            }
            var next = nextBlockSibling(block);
            while (next && next.matches(OUTPUT_SELECTOR)) {
                var nextPre = preOf(next);
                if (!nextPre) {
                    break;
                }
                targetPre.appendChild(document.createTextNode("\n"));
                while (nextPre.firstChild) {
                    targetPre.appendChild(nextPre.firstChild);
                }
                var following = nextBlockSibling(next);
                next.dataset.pqMerged = "into";
                next.parentNode.removeChild(next);
                next = following;
            }
        });
    }

    // Wrap tall output blocks with a Show more / Show less toggle.
    function makeCollapsible(root) {
        var blocks = root.querySelectorAll(OUTPUT_SELECTOR);
        blocks.forEach(function (block) {
            if (block.dataset.pqProcessed === "true") {
                return;
            }
            block.dataset.pqProcessed = "true";

            if (block.scrollHeight <= COLLAPSE_PX) {
                return;
            }

            block.classList.add("pq-collapsed");

            var button = document.createElement("button");
            button.type = "button";
            button.className = "pq-show-more";
            button.textContent = "Show more";
            button.addEventListener("click", function () {
                var collapsed = block.classList.toggle("pq-collapsed");
                button.textContent = collapsed ? "Show more" : "Show less";
            });

            block.parentNode.insertBefore(button, block.nextSibling);
        });
    }

    function process() {
        var root = document.querySelector("main") || document.body;
        if (!root) {
            return;
        }
        mergeConsecutiveOutputs(root);
        makeCollapsible(root);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", process);
    } else {
        process();
    }
})();
