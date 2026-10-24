/* Runs in the head so the selected appearance is applied before first paint. */
(() => {
  'use strict';
  const key = 'sevenwonders-dashboard-theme';
  let preference = null;
  try {
    const saved = localStorage.getItem(key);
    if (saved === 'light' || saved === 'dark') preference = saved;
  } catch (_) { /* Local-file storage can be unavailable; keep the toggle usable. */ }
  const media = typeof matchMedia === 'function' ? matchMedia('(prefers-color-scheme: dark)') : null;
  let theme = preference || (media?.matches ? 'dark' : 'light');
  let toggle = null;
  const t = key => globalThis.SevenWondersI18n?.t(key)??key;
  function apply() {
    document.documentElement.setAttribute('data-theme', theme);
    if (toggle) {
      toggle.textContent = t(theme === 'dark' ? 'Dark Mode: an' : 'Dark Mode: aus');
      toggle.setAttribute('aria-pressed', String(theme === 'dark'));
      toggle.title = t(theme === 'dark' ? 'Zu heller Darstellung wechseln' : 'Zu dunkler Darstellung wechseln');
    }
  }
  media?.addEventListener?.('change', event => {
    if (!preference) { theme = event.matches ? 'dark' : 'light'; apply(); }
  });
  globalThis.SevenWondersTheme = {
    attach(button) {
      if (!button) return;
      toggle = button;
      toggle.addEventListener('click', () => {
        theme = theme === 'dark' ? 'light' : 'dark';
        preference = theme;
        try { localStorage.setItem(key, theme); } catch (_) { /* Use session preference. */ }
        apply();
      });
      apply();
    }
  };
  globalThis.SevenWondersI18n?.subscribe(apply);
  apply();
})();
