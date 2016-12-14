window.addEventListener('load', function() {
  startRes();
}, false);
function startRes() {
  TagCanvas.Start('resCanvas', 'demoTags', {
    fadeIn:500,
    textColour: '#000',
    textHeight: 25,
    maxSpeed: 0.02,
    minBrightness: 0.2,
    depth: 0.5,
    pulsateTo: 0.6,
    initial: [0.03,-0.03],
    decel: 0.98,
    reverse: true,
    imageScale: null,
    fadeIn: 1000,
    clickToFront: 600,
    pulsateTo: 0.2,
    pulsateTime: 0.5,
    outlineMethod: 'none',
    outlineColour: 'none',
    lock: 'x',
    shape: 'hcylinder',
    radiusX: 2.5,
    wheelZoom: 0
  });
}
function badDims() {
  var c = document.getElementById('resCanvas');
  c.width = 170;
  c.height = 75;
  startRes();
}