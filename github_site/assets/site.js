(function(){
  var root=document.documentElement,KEY='theme';
  function store(v){try{if(v)localStorage.setItem(KEY,v);else localStorage.removeItem(KEY)}catch(e){}}
  function saved(){try{return localStorage.getItem(KEY)}catch(e){return null}}
  var s=saved(); if(s==='light'||s==='dark')root.setAttribute('data-theme',s);
  function isDark(){var t=root.getAttribute('data-theme');return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches}
  var tb=document.getElementById('themeBtn');
  function label(){tb.querySelector('span').textContent=isDark()?'Light':'Dark';tb.setAttribute('aria-label','Switch to '+(isDark()?'light':'dark')+' theme')}
  tb.addEventListener('click',function(){var n=isDark()?'light':'dark';root.setAttribute('data-theme',n);store(n);label()});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',label);label();

  var body=document.body,mb=document.getElementById('menuBtn');
  mb.addEventListener('click',function(){var o=body.classList.toggle('menu');mb.setAttribute('aria-expanded',o)});
  document.querySelectorAll('.toc a').forEach(function(a){a.addEventListener('click',function(){body.classList.remove('menu');mb.setAttribute('aria-expanded','false')})});
  addEventListener('keydown',function(e){if(e.key==='Escape')body.classList.remove('menu')});

  var links=[].slice.call(document.querySelectorAll('.toc a')),secs=links.map(function(a){return document.querySelector(a.getAttribute('href'))});
  var now=document.getElementById('now'),bar=document.getElementById('progress'),top=document.getElementById('top'),tick=false;
  function update(){
    tick=false;var y=scrollY,h=document.documentElement.scrollHeight-innerHeight,cur=0;
    bar.style.transform='scaleX('+(h>0?Math.min(1,y/h):0)+')';
    for(var i=0;i<secs.length;i++){if(secs[i]&&secs[i].getBoundingClientRect().top<120)cur=i}
    links.forEach(function(a,i){a.classList.toggle('on',i===cur);if(i===cur)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current')});
    now.textContent=links[cur]?links[cur].textContent:'';
    top.classList.toggle('show',y>900);
  }
  addEventListener('scroll',function(){if(!tick){tick=true;requestAnimationFrame(update)}},{passive:true});
  addEventListener('resize',update);update();
  top.addEventListener('click',function(){scrollTo({top:0})});
})();
