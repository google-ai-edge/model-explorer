/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

/** Scrollbar geometry and interaction adapted from candidates updateConversationScrollbar.
 * Keeps the reading grid full-width without removing pointer/keyboard scrolling.
 */
export function createConversationScrollbar(
  root: HTMLElement,
  isEnabled: () => boolean,
) {
  const bar = document.createElement('div');
  bar.className = 'conversationScrollbar';
  bar.tabIndex = 0;
  bar.setAttribute('role', 'scrollbar');
  bar.setAttribute('aria-label', 'Scroll both conversations');
  bar.setAttribute('aria-orientation', 'vertical');
  root.append(bar);
  let sc: HTMLElement | null = null,
    frame = 0,
    timer = 0;
  const abort = new AbortController();
  const update = () => {
    frame = 0;
    if (!sc || !isEnabled()) {
      bar.hidden = true;
      return;
    }
    const wr = root.getBoundingClientRect(),
      sr = sc.getBoundingClientRect(),
      dock = root.querySelector('.composer')!.getBoundingClientRect();
    const head =
      sc.querySelector('.column-headings')?.getBoundingClientRect().height ??
      60;
    const top = sr.top + head,
      height = Math.max(40, Math.min(sr.bottom, dock.top - 12) - top - 8),
      max = Math.max(0, sc.scrollHeight - sc.clientHeight),
      thumb = Math.min(
        height,
        Math.max(30, (height * sc.clientHeight) / sc.scrollHeight),
      );
    bar.style.left = sr.right - wr.left - 12 + 'px';
    bar.style.top = top - wr.top + 4 + 'px';
    bar.style.height = height + 'px';
    bar.style.setProperty('--thumb-height', thumb + 'px');
    bar.style.setProperty(
      '--thumb-top',
      (max ? ((height - thumb) * sc.scrollTop) / max : 0) + 'px',
    );
    sc.id = 'conversationReading';
    bar.setAttribute('aria-controls', sc.id);
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', String(Math.round(max)));
    bar.setAttribute('aria-valuenow', String(Math.round(sc.scrollTop)));
    bar.hidden = max === 0;
  };
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(update);
  };
  const active = () => {
    bar.classList.add('visible');
    clearTimeout(timer);
    timer = window.setTimeout(() => bar.classList.remove('visible'), 1000);
    schedule();
  };
  bar.addEventListener('keydown', (e) => {
    if (!sc) return;
    const amount: Record<string, number> = {
      ArrowDown: 40,
      ArrowUp: -40,
      PageDown: sc.clientHeight * 0.8,
      PageUp: -sc.clientHeight * 0.8,
    };
    if (e.key in amount) sc.scrollTop += amount[e.key];
    else if (e.key === 'Home') sc.scrollTop = 0;
    else if (e.key === 'End') sc.scrollTop = sc.scrollHeight;
    else return;
    e.preventDefault();
    active();
  });
  bar.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || !sc) return;
    e.preventDefault();
    bar.focus({preventScroll: true});
    bar.setPointerCapture(e.pointerId);
    bar.classList.add('dragging');
    const rect = bar.getBoundingClientRect(),
      thumb = parseFloat(bar.style.getPropertyValue('--thumb-height')),
      top = parseFloat(bar.style.getPropertyValue('--thumb-top')),
      local = e.clientY - rect.top,
      offset = local >= top && local <= top + thumb ? local - top : thumb / 2;
    const move = (event: PointerEvent) => {
      if (sc)
        sc.scrollTop =
          Math.max(
            0,
            Math.min(
              1,
              (event.clientY - rect.top - offset) /
                Math.max(1, rect.height - thumb),
            ),
          ) *
          (sc.scrollHeight - sc.clientHeight);
      update();
    };
    move(e);
    bar.onpointermove = move;
    bar.onlostpointercapture = () => {
      bar.onpointermove = null;
      bar.classList.remove('dragging');
    };
  });
  const observer = new ResizeObserver(schedule);
  return {
    mount() {
      const next = root.querySelector<HTMLElement>('.reading');
      if (next !== sc) {
        sc = next;
        if (sc) {
          sc.addEventListener('scroll', active, {
            passive: true,
            signal: abort.signal,
          });
          sc.addEventListener(
            'pointermove',
            (e) => {
              if (e.clientX > sc!.getBoundingClientRect().right - 20) active();
            },
            {passive: true, signal: abort.signal},
          );
          observer.observe(sc);
          const turns = sc.querySelector('.turns');
          if (turns) observer.observe(turns);
          observer.observe(root);
        }
      }
      schedule();
    },
    destroy() {
      abort.abort();
      observer.disconnect();
      cancelAnimationFrame(frame);
      clearTimeout(timer);
      bar.remove();
    },
  };
}
