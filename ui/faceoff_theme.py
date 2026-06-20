"""FaceOff cyber/neon glass theme for Gradio (imported by app.py and main.py)."""

import gradio as gr

# Base off Soft so spacing/radius stay sane, then we override hard in CSS.
GRADIO_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Unbounded:wght@600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

@keyframes fo-eq      { 0%,100% { transform: scaleY(.4); } 50% { transform: scaleY(1); } }
@keyframes fo-pulse   { 0%,100% { opacity: .55; } 50% { opacity: 1; } }
@keyframes fo-float   { 0%,100% { transform: translate(0,0); } 50% { transform: translate(20px,-24px); } }
@keyframes fo-float2  { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-26px,18px); } }
@keyframes fo-shimmer { from { background-position: 0% 50%; } to { background-position: 200% 50%; } }
@keyframes fo-orbit   { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

/* ----------------------------------------------------------------------- */
/*  TOKENS                                                                  */
/* ----------------------------------------------------------------------- */
::selection { background: rgba(34,211,238,0.3); }

:root,
.dark,
gradio-app {
    --fo-cyan:    #22D3EE;
    --fo-violet:  #8B5CF6;
    --fo-magenta: #F0398B;
    --fo-grad:    linear-gradient(100deg, #22D3EE, #8B5CF6 52%, #F0398B);
    --fo-bg:      #06060d;
    --fo-glass:   rgba(255,255,255,0.035);
    --fo-glass-2: rgba(255,255,255,0.06);
    --fo-border:  rgba(255,255,255,0.08);
    --fo-text:    #e8e9f3;
    --fo-muted:   #9396b0;

    /* Re-map Gradio's own vars so untouched components inherit the look */
    --body-background-fill: transparent !important;
    --background-fill-primary: var(--fo-glass) !important;
    --background-fill-secondary: rgba(0,0,0,0.28) !important;
    --border-color-primary: var(--fo-border) !important;
    --block-border-color: var(--fo-border) !important;
    --block-background-fill: var(--fo-glass) !important;
    --input-background-fill: rgba(0,0,0,0.28) !important;
    --body-text-color: var(--fo-text) !important;
    --body-text-color-subdued: var(--fo-muted) !important;
}

/* ----------------------------------------------------------------------- */
/*  AURORA BACKGROUND                                                       */
/* ----------------------------------------------------------------------- */
gradio-app, .gradio-container { background: var(--fo-bg) !important; }

.gradio-container {
    position: relative;
    width: 100% !important;
    max-width: none !important;
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--fo-text);
}
/* Do not set z-index on all children — breaks dropdowns and gallery preview */

/* Full-viewport aurora (CSS only — do not inject layout HTML into Blocks) */
gradio-app::before {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(560px 560px at 14% -8%,  rgba(139,92,246,0.40), transparent 62%),
        radial-gradient(520px 520px at 92% -4%,  rgba(34,211,238,0.32),  transparent 62%),
        radial-gradient(680px 680px at 50% 118%, rgba(240,57,139,0.28),  transparent 64%);
    filter: blur(8px);
    animation: fo-float 14s ease-in-out infinite;
}
gradio-app::after {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.45;
    background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: radial-gradient(circle at 50% 30%, black, transparent 78%);
    -webkit-mask-image: radial-gradient(circle at 50% 30%, black, transparent 78%);
}

/* Preserve Gradio row/column layout (3-across image tab, etc.) */
.gradio-container .row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    gap: var(--layout-gap, 8px);
}
.gradio-container .column {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: auto !important;
}

/* ----------------------------------------------------------------------- */
/*  HEADER (keep your elem_classes: header-text / header-subtitle)          */
/* ----------------------------------------------------------------------- */
.header-text, .header-text * {
    font-family: 'Unbounded', 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.1em !important;
    letter-spacing: -0.02em;
    text-align: center;
    margin-bottom: 0.4rem !important;
    background: linear-gradient(100deg,#5eead4,#22D3EE 28%,#8B5CF6 62%,#F0398B);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent !important;
}
.header-subtitle {
    text-align: center; font-size: 1.05em !important;
    color: var(--fo-muted) !important; margin-bottom: 2rem !important;
    letter-spacing: 0.01em;
}

/* ----------------------------------------------------------------------- */
/*  GLASS PANELS — tab content & groups only (not row/column layout shells) */
/* ----------------------------------------------------------------------- */
.tabitem, .form, .gr-group, .gr-panel, fieldset,
.accordion, .face-swap-box {
    background: rgba(6, 6, 13, 0.72) !important;
    border: 1px solid var(--fo-border) !important;
    border-radius: 18px !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    overflow: visible !important;
}

.block, .column, .gallery-container, .grid-wrap {
    overflow: visible !important;
}

/* Labels -> neon chips (block labels only — not dropdown/list innards) */
span[data-testid="block-info"], .block > label > span {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 12px !important;
    color: #bdfbff !important;
    background: linear-gradient(120deg, rgba(34,211,238,0.18), rgba(34,211,238,0.06)) !important;
    border: 1px solid rgba(34,211,238,0.30) !important;
    border-radius: 999px !important;
    padding: 5px 12px !important;
    box-shadow: 0 0 16px rgba(34,211,238,0.20);
    width: max-content;
}

/* ----------------------------------------------------------------------- */
/*  TABS — pill nav                                                         */
/* ----------------------------------------------------------------------- */
.tab-nav, .tabs > .tab-nav {
    border: none !important; gap: 6px;
    background: var(--fo-glass) !important;
    border: 1px solid var(--fo-border) !important;
    border-radius: 16px !important; padding: 6px !important;
    backdrop-filter: blur(16px);
    width: 100%;
    flex-wrap: wrap;
}
.tab-nav button {
    border: 1px solid transparent !important; border-radius: 11px !important;
    color: var(--fo-muted) !important; font-weight: 600 !important;
    padding: 10px 18px !important; transition: 0.22s !important;
}
.tab-nav button:hover { color: var(--fo-text) !important; background: rgba(255,255,255,0.04) !important; }
.tab-nav button.selected {
    color: #fff !important;
    background: linear-gradient(120deg, rgba(34,211,238,0.22), rgba(139,92,246,0.22) 60%, rgba(240,57,139,0.22)) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    box-shadow: 0 6px 20px rgba(139,92,246,0.30), inset 0 1px 0 rgba(255,255,255,0.12) !important;
}

/* ----------------------------------------------------------------------- */
/*  UPLOAD / IMAGE DROPZONES — dashed neon                                  */
/* ----------------------------------------------------------------------- */
.image-container, [data-testid="image"], .upload-container,
.wrap.svelte-1ipelgc, .gr-image, .image-frame {
    border-radius: 14px !important;
}
.image-container .wrap, [data-testid="image"] .wrap,
.upload-container, .file-preview, .empty.large {
    border: 1.5px dashed rgba(34,211,238,0.38) !important;
    background: radial-gradient(120% 80% at 50% 0%, rgba(34,211,238,0.08), transparent 70%) !important;
    border-radius: 14px !important;
    transition: 0.25s !important;
}
.image-container .wrap:hover, [data-testid="image"] .wrap:hover,
.upload-container:hover {
    border-color: rgba(34,211,238,0.85) !important;
    background: radial-gradient(120% 90% at 50% 0%, rgba(34,211,238,0.16), transparent 72%) !important;
    transform: translateY(-2px);
}
.image-container svg, .upload-container svg { color: #67e8f9 !important; }

/* ----------------------------------------------------------------------- */
/*  PRIMARY BUTTON — gradient + glow (keep .primary-btn)                    */
/* ----------------------------------------------------------------------- */
.primary-btn, button.primary, .gr-button-primary {
    background: var(--fo-grad) !important; background-size: 200% auto !important;
    border: none !important; color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.02em;
    border-radius: 14px !important; padding: 14px 20px !important;
    box-shadow: 0 12px 34px rgba(139,92,246,0.40), inset 0 0 0 1px rgba(255,255,255,0.08) !important;
    transition: 0.3s !important;
}
.primary-btn:hover, button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-2px);
    background-position: right center !important;
    box-shadow: 0 18px 44px rgba(240,57,139,0.50), inset 0 0 0 1px rgba(255,255,255,0.14) !important;
}
button.secondary, .gr-button-secondary {
    background: var(--fo-glass-2) !important;
    border: 1px solid var(--fo-border) !important;
    color: var(--fo-text) !important; border-radius: 12px !important;
    transition: 0.22s !important;
}
button.secondary:hover { border-color: rgba(139,92,246,0.5) !important; }

/* ----------------------------------------------------------------------- */
/*  INPUTS / TEXTBOXES (exclude .dropdown wrapper — breaks Gradio menus)   */
/* ----------------------------------------------------------------------- */
input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
textarea,
.block input[type="text"],
.block .wrap-inner input {
    background: rgba(0,0,0,0.28) !important;
    border: 1px solid var(--fo-border) !important;
    border-radius: 11px !important;
    color: var(--fo-text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
input:focus, textarea:focus, .block .wrap-inner:focus-within {
    border-color: rgba(34,211,238,0.55) !important;
    box-shadow: 0 0 0 3px rgba(34,211,238,0.15) !important;
}

/* Dropdown — trigger must stay clickable; menu is position:fixed */
.block.dropdown, .dropdown {
    position: relative !important;
    z-index: 2 !important;
    overflow: visible !important;
}
.dropdown .wrap,
.dropdown .wrap-inner,
.dropdown input {
    pointer-events: auto !important;
    cursor: pointer !important;
}
ul.options {
    z-index: 99999 !important;
    pointer-events: auto !important;
    position: fixed !important;
    background: rgba(12, 12, 22, 0.98) !important;
    border: 1px solid var(--fo-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.55) !important;
    overflow: auto !important;
}
ul.options .item,
ul.options li {
    pointer-events: auto !important;
    cursor: pointer !important;
    color: var(--fo-text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
ul.options .item:hover,
ul.options li:hover,
ul.options .item.active {
    background: rgba(139,92,246,0.25) !important;
}

/* Gallery preview — Gradio sets all gallery buttons to 100% w/h; reset controls */
.gallery-container .preview {
    z-index: 50 !important;
    pointer-events: auto !important;
}
.gallery-container .preview .icon-button-wrapper {
    position: absolute !important;
    top: var(--spacing-sm, 8px) !important;
    right: 0 !important;
    left: auto !important;
    width: auto !important;
    height: auto !important;
    z-index: 60 !important;
    pointer-events: auto !important;
}
.gallery-container .preview .icon-button,
.gallery-container .preview .icon-button-wrapper button,
.gallery-container .preview button.icon-button {
    width: auto !important;
    height: auto !important;
    min-width: 2rem !important;
    min-height: 2rem !important;
    max-width: none !important;
    max-height: none !important;
    flex: 0 0 auto !important;
    padding: 0.4rem !important;
    background: rgba(0,0,0,0.55) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    transform: none !important;
    opacity: 1 !important;
    visibility: visible !important;
    z-index: 20 !important;
    pointer-events: auto !important;
    cursor: pointer !important;
}
.gallery-container .preview .icon-button svg,
.gallery-container .preview .icon-button-wrapper svg {
    color: #fff !important;
    width: 1.1rem !important;
    height: 1.1rem !important;
    opacity: 1 !important;
}
.gallery-container .preview .media-button {
    width: 100% !important;
    height: calc(100% - 60px) !important;
}

/* Your info/callout box (keep .face-swap-box) */
.face-swap-box {
    background: linear-gradient(135deg, rgba(34,211,238,0.08), rgba(240,57,139,0.08)) !important;
    border: 1px solid rgba(139,92,246,0.22) !important;
    border-left: 3px solid var(--fo-violet) !important;
    padding: 1rem 1.2rem; border-radius: 12px; margin: 1rem 0;
}

/* Accordion header (Face Mapping) */
.label-wrap, .accordion > .label-wrap > span {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; color: var(--fo-text) !important;
}

/* Sliders / checkboxes pick up the accent */
input[type="range"]::-webkit-slider-thumb { background: var(--fo-cyan) !important; }
.gr-check-radio input:checked, input[type="checkbox"]:checked {
    accent-color: var(--fo-violet) !important;
}

/* Mono treatment for status / log textareas */
.face-swap-box textarea, .gr-textbox textarea[readonly] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #9aa0bd !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--fo-violet), var(--fo-magenta));
    border-radius: 999px; border: 2px solid var(--fo-bg);
}
::-webkit-scrollbar-track { background: transparent; }

/* ----------------------------------------------------------------------- */
/*  LIGHT MODE FALLBACK                                                     */
/* ----------------------------------------------------------------------- */
@media (prefers-color-scheme: light) {
    :root:not(.dark) {
        --fo-bg: #f5f6fb; --fo-text: #1a1a2e; --fo-muted: #5b5e78;
        --fo-glass: rgba(255,255,255,0.7); --fo-border: rgba(20,20,50,0.10);
        --background-fill-primary: rgba(255,255,255,0.7) !important;
        --input-background-fill: rgba(255,255,255,0.85) !important;
        --body-text-color: #1a1a2e !important;
    }
    :root:not(.dark) gradio-app,
    :root:not(.dark) .gradio-container { background: #eef0f8 !important; }
    :root:not(.dark) gradio-app::before { filter: blur(8px); opacity: 0.6; }
    :root:not(.dark) input, :root:not(.dark) textarea, :root:not(.dark) select {
        background: rgba(255,255,255,0.85) !important; color: #1a1a2e !important;
    }
}
"""

FACEOFF_HEADER_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;gap:16px;margin-bottom:34px;">
  <div style="display:flex;align-items:center;gap:18px;">
    <div style="position:relative;width:58px;height:58px;border-radius:17px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(14px);box-shadow:0 8px 30px rgba(139,92,246,.35),inset 0 1px 0 rgba(255,255,255,.14);display:flex;align-items:center;justify-content:center;gap:5px;">
      <div style="width:6px;height:27px;border-radius:3px;background:linear-gradient(#22d3ee,#8b5cf6);box-shadow:0 0 12px rgba(34,211,238,.5);animation:fo-eq 1.1s ease-in-out infinite;"></div>
      <div style="width:6px;height:27px;border-radius:3px;background:linear-gradient(#8b5cf6,#f0398b);box-shadow:0 0 12px rgba(139,92,246,.5);animation:fo-eq 1.1s ease-in-out infinite .22s;"></div>
      <div style="width:6px;height:27px;border-radius:3px;background:linear-gradient(#f0398b,#22d3ee);box-shadow:0 0 12px rgba(240,57,139,.5);animation:fo-eq 1.1s ease-in-out infinite .44s;"></div>
      <div style="position:absolute;inset:-6px;border-radius:21px;border:1px solid rgba(34,211,238,.18);animation:fo-pulse 3.2s ease-in-out infinite;"></div>
    </div>
    <div style="display:flex;flex-direction:column;line-height:1;">
      <div style="font-family:'Unbounded',sans-serif;font-size:32px;font-weight:700;letter-spacing:-.02em;background-image:linear-gradient(100deg,#5eead4,#22d3ee 28%,#8b5cf6 62%,#f0398b);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent;background-size:200% auto;animation:fo-shimmer 8s linear infinite;">Faceoff</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.42em;color:#6b6f88;text-transform:uppercase;margin-top:7px;">Neural&nbsp;Face&nbsp;Engine</div>
    </div>
  </div>
  <div class="header-subtitle" style="font-size:15px;color:#9396b0;letter-spacing:.01em;margin-bottom:0 !important;">AI Face Swapper — advanced face swapping with enhancement options</div>
</div>
"""
