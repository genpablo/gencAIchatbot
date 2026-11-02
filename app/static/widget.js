/* Gen C Beauty assistant: chat bubble for the online shop (BigCommerce).
   Add it in BigCommerce admin > Storefront > Script Manager, on all pages, in the footer:
     <script src="https://YOUR-APP-ADDRESS/widget.js" defer></script>
   Optional attributes: data-label="Ask us" (bubble text), data-position="left" (default: right).
   The chat runs in a frame from this app (/shop?embed=1); product links open in the shop window. */
(function () {
  if (window.__gencAssistant) return;
  window.__gencAssistant = true;
  var script = document.currentScript || document.querySelector('script[src*="/widget.js"]');
  if (!script) return;
  var origin = new URL(script.src, location.href).origin;
  var label = script.getAttribute("data-label") || "Ask us";
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";

  var css = document.createElement("style");
  css.textContent =
    ".gca-btn{position:fixed;bottom:20px;" + side + ":20px;z-index:2147483000;display:flex;align-items:center;gap:8px;" +
    "height:52px;padding:0 20px 0 16px;border:0;border-radius:26px;background:#0F0F2D;color:#fff;cursor:pointer;" +
    "font:800 13px/1 Nunito,Arial,Helvetica,sans-serif;letter-spacing:.04em;text-transform:uppercase;box-shadow:0 6px 20px rgba(15,15,45,.3)}" +
    ".gca-btn svg{width:22px;height:22px}" +
    ".gca-btn:focus-visible{outline:3px solid #1F439E;outline-offset:2px}" +
    ".gca-panel{position:fixed;bottom:84px;" + side + ":20px;z-index:2147483001;width:400px;height:min(640px,calc(100vh - 110px));" +
    "background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 12px 40px rgba(15,15,45,.3);display:none}" +
    ".gca-panel.gca-open{display:block}" +
    ".gca-panel iframe{width:100%;height:100%;border:0;display:block}" +
    "@media (max-width:520px){.gca-panel{inset:0;width:auto;height:auto;border-radius:0;bottom:0}" +
    ".gca-open~.gca-btn{display:none}}";
  document.head.appendChild(css);

  var panel = document.createElement("div");
  panel.className = "gca-panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Gen C Beauty assistant");
  var btn = document.createElement("button");
  btn.type = "button";
  btn.className = "gca-btn";
  btn.setAttribute("aria-expanded", "false");
  var chatIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 5h16v11H9l-5 4z"/></svg>';
  var closeIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M6 6l12 12M18 6L6 18"/></svg>';
  function paint(open) {
    btn.innerHTML = (open ? closeIcon : chatIcon) + "<span></span>";
    btn.lastChild.textContent = open ? "Close" : label;
    btn.setAttribute("aria-label", open ? "Close the assistant" : "Open the assistant");
  }
  paint(false);

  var frame = null;
  function open() {
    if (!frame) {                       // load the chat only when a customer first opens it
      frame = document.createElement("iframe");
      frame.title = "Gen C Beauty assistant";
      frame.src = origin + "/shop?embed=1";
      panel.appendChild(frame);
    }
    panel.classList.add("gca-open");
    btn.setAttribute("aria-expanded", "true");
    paint(true);
  }
  function close() {
    panel.classList.remove("gca-open");
    btn.setAttribute("aria-expanded", "false");
    paint(false);
    btn.focus();
  }
  btn.onclick = function () { panel.classList.contains("gca-open") ? close() : open(); };
  window.addEventListener("message", function (e) {          // the chat's own close button (phones)
    if (e.origin === origin && e.data && e.data.type === "gca:close") close();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && panel.classList.contains("gca-open")) close();
  });
  document.body.appendChild(panel);
  document.body.appendChild(btn);
})();
