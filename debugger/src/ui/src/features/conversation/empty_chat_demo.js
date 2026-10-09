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

/**
 * Empty illustration controller inherited from candidates
 * conversation-step-zero.html. Timeline unchanged; scoped to the Angular host
 * and cleaned up on destroy.
 */
export function mountEmptyChatDemo(root, demo) {
  const input = root.querySelector('textarea');
  if (!input) return () => {};
  const events = new AbortController(), signal = events.signal;
  const demoTiming = {
    rounds: [{input: 0.85, transfer: 0.18, reply: 1.6, targetLag: 0.3}],
    hold: 2,
    fade: 0.5,
  };
  function configureTimeline() {
    if (!demo) return;
    const first = demoTiming.rounds[0], send = first.input,
          replyStart = send + first.transfer;
    const fadeStart =
        replyStart + first.reply + first.targetLag + demoTiming.hold;
    const total = fadeStart + demoTiming.fade,
          pct = (t) => `${((100 * t) / total).toFixed(6)}%`;
    demo.style.setProperty('--demo-cycle', `${total}s`);
    const css = [
      `@keyframes demoGroup{0%,${pct(fadeStart)}{opacity:1}100%{opacity:0}}`
    ];
    let characterId = 0;
    function typeCharacters(element, start, duration) {
      const nodes = [],
            walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) nodes.push(walker.currentNode);
      const count =
          nodes.reduce((sum, n) => sum + Array.from(n.textContent).length, 0);
      let index = 0;
      nodes.forEach((node) => {
        const fragment = document.createDocumentFragment();
        for (const character of Array.from(node.textContent)) {
          const span = document.createElement('span'),
                name = 'demoCharacter' + characterId++,
                time = start + (++index / count) * duration;
          span.className = 'demoCharacter';
          span.textContent = character;
          span.style.animationName = name;
          css.push(
              `@keyframes ${name}{0%,${
                  pct(Math.max(0, time - 0.001))}{visibility:hidden}${
                  pct(time)},100%{visibility:visible}}`,
          );
          fragment.append(span);
        }
        node.replaceWith(fragment);
      });
    }
    typeCharacters(demo.querySelector('.demoDraftText'), 0.05, 0.6);
    demo.querySelectorAll('.demoAnswer')
        .forEach(
            (answer, side) => typeCharacters(
                answer,
                replyStart + 0.1,
                (first.reply + (side ? first.targetLag : 0) - 0.1) / 2,
                ),
        );
    css.push(
        `@keyframes demoDraftExample{0%,${pct(send)}{opacity:1}${
            pct(send + 0.08)},100%{opacity:0}}`,
    );
    css.push(
        `@keyframes demoPrompt{0%,${
            pct(send + 0.08)}{opacity:0;transform:none}${
            pct(replyStart)},100%{opacity:1;transform:none}}`,
    );
    css.push(
        `@keyframes demoModelLabel{0%,${pct(replyStart)}{opacity:0}${
            pct(replyStart + 0.15)},100%{opacity:1}}`,
    );
    const style = demo.querySelector('style[data-demo-timeline]') ||
        document.createElement('style');
    style.dataset.demoTimeline = '';
    style.textContent = css.join('\n');
    if (!style.isConnected) demo.append(style);
  }
  configureTimeline();

  let composing = false, withdrawTimer;
  function positionDemo() {
    if (!demo?.isConnected) return;
    const
        r = input.getBoundingClientRect(),
        example = demo.querySelector('.demoDraftText'), is = getComputedStyle(input);
    Object.assign(example.style, {
      left: r.x + parseFloat(is.paddingLeft) + 'px',
      top: r.y + parseFloat(is.paddingTop) + 'px',
      width: r.width - parseFloat(is.paddingLeft) -
          parseFloat(is.paddingRight) + 'px',
      font: is.font,
      lineHeight: is.lineHeight,
    });
    demo.style.setProperty(
        '--stage-height',
        `${
            root.querySelector('.composer').getBoundingClientRect().top -
            demo.getBoundingClientRect().top - 8}px`,
    );
  }
  function syncDraft() {
    if (!demo?.isConnected) return;
    const hide = composing || document.activeElement === input ||
        root.querySelector('.mode-switch').contains(document.activeElement) ||
        input.value.length > 0 || false;
    const wasHidden = demo.classList.contains('withdrawn');
    if (hide && !wasHidden) {
      clearTimeout(withdrawTimer);
      withdrawTimer = setTimeout(() => demo.classList.add('demoReset'), 220);
    } else if (!hide && wasHidden) {
      clearTimeout(withdrawTimer);
      demo.classList.add('demoReset');
      void demo.offsetWidth;
      demo.classList.remove('demoReset');
    }
    demo.classList.toggle('withdrawn', hide);
    root.classList.toggle('demoRunning', !hide);
    positionDemo();
  }
  input.addEventListener('input', syncDraft, {signal});
  input.addEventListener('focus', syncDraft, {signal});
  input.addEventListener(
      'blur', () => requestAnimationFrame(syncDraft), {signal});
  const modes = root.querySelector('.mode-switch');
  modes.addEventListener('focusin', syncDraft, {signal});
  modes.addEventListener(
      'focusout', () => requestAnimationFrame(syncDraft), {signal});
  modes.addEventListener('click', syncDraft, {signal});
  input.addEventListener(
      'compositionstart',
      () => {
        composing = true;
        syncDraft();
      },
      {signal},
  );
  input.addEventListener(
      'compositionend',
      () => {
        composing = false;
        syncDraft();
      },
      {signal},
  );
  const observer = new ResizeObserver(positionDemo);
  observer.observe(root.querySelector('.composer'));
  window.addEventListener('resize', positionDemo, {signal});
  document.fonts.ready.then(positionDemo);
  syncDraft();

  return () => {
    events.abort();
    observer.disconnect();
    clearTimeout(withdrawTimer);
    root.classList.remove('demoRunning');
  };
}
