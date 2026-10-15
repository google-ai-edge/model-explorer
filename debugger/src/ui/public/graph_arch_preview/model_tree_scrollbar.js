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

// Match conversation_scrollbar.ts interaction and _conversation_virtualization.scss geometry.
window.createModelTreeScrollbar = function (scroller) {
  const root = scroller.parentElement;
  const bar = document.createElement('div');
  bar.className = 'model-tree-scrollbar';
  bar.tabIndex = 0;
  bar.setAttribute('role', 'scrollbar');
  bar.setAttribute('aria-label', 'Scroll model modules');
  bar.setAttribute('aria-orientation', 'vertical');
  bar.setAttribute('aria-controls', scroller.id);
  root.append(bar);
  let frame = 0, timer = 0;
  function update() {
    frame = 0;
    const height = Math.max(0, scroller.clientHeight - 8);
    const max = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
    const thumb = Math.min(height, Math.max(30, height * scroller.clientHeight / Math.max(1, scroller.scrollHeight)));
    bar.style.left = (scroller.offsetLeft + scroller.offsetWidth - 12) + 'px';
    bar.style.top = (scroller.offsetTop + 4) + 'px';
    bar.style.height = height + 'px';
    bar.style.setProperty('--thumb-height', thumb + 'px');
    bar.style.setProperty('--thumb-top', (max ? (height - thumb) * scroller.scrollTop / max : 0) + 'px');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', String(Math.round(max)));
    bar.setAttribute('aria-valuenow', String(Math.round(scroller.scrollTop)));
    bar.hidden = max === 0 || height === 0;
  }
  function schedule() { if (!frame) frame = requestAnimationFrame(update); }
  function active() {
    bar.classList.add('visible');
    clearTimeout(timer);
    timer = setTimeout(() => bar.classList.remove('visible'), 1000);
    schedule();
  }
  scroller.addEventListener('scroll', active, {passive: true});
  scroller.addEventListener('pointermove', e => {
    if (e.clientX > scroller.getBoundingClientRect().right - 20) active();
  }, {passive: true});
  bar.addEventListener('keydown', e => {
    const amount = {ArrowDown: 40, ArrowUp: -40, PageDown: scroller.clientHeight * .8, PageUp: -scroller.clientHeight * .8};
    if (e.key in amount) scroller.scrollTop += amount[e.key];
    else if (e.key === 'Home') scroller.scrollTop = 0;
    else if (e.key === 'End') scroller.scrollTop = scroller.scrollHeight;
    else return;
    e.preventDefault(); active();
  });
  bar.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    e.preventDefault(); bar.focus({preventScroll: true}); bar.setPointerCapture(e.pointerId);
    bar.classList.add('dragging');
    const rect = bar.getBoundingClientRect(), thumb = parseFloat(bar.style.getPropertyValue('--thumb-height'));
    const top = parseFloat(bar.style.getPropertyValue('--thumb-top')), local = e.clientY - rect.top;
    const offset = local >= top && local <= top + thumb ? local - top : thumb / 2;
    const move = event => {
      scroller.scrollTop = Math.max(0, Math.min(1, (event.clientY - rect.top - offset) / Math.max(1, rect.height - thumb))) * (scroller.scrollHeight - scroller.clientHeight);
      update();
    };
    move(e); bar.onpointermove = move;
    bar.onlostpointercapture = () => { bar.onpointermove = null; bar.classList.remove('dragging'); active(); };
  });
  new ResizeObserver(schedule).observe(scroller);
  new MutationObserver(schedule).observe(scroller, {childList: true, subtree: true, attributes: true, attributeFilter: ['hidden']});
  schedule();
};
