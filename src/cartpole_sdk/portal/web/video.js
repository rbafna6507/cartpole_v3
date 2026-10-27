/* Read-only length-prefixed Annex-B video. No camera settings or rig commands. */
(() => {
  const MAX_PACKET_BYTES = 8 * 1024 * 1024;
  async function waitCapacity(decoder, signal) {
    if (signal.aborted || decoder.state === 'closed') throw Error('Video stopped');
    const start = performance.now();
    while (decoder.decodeQueueSize >= 6) {
      if (signal.aborted || decoder.state === 'closed') throw Error('Video stopped');
      if (performance.now() - start > 250) throw Error('Video decoder stalled');
      await new Promise(resolve => setTimeout(resolve, 4));
    }
  }
  class PortalVideo {
    constructor(canvas, callbacks = {}) {
      this.canvas = canvas; this.context = canvas.getContext('2d', {alpha: false}); this.callbacks = callbacks;
      this.generation = 0; this.controller = null; this.decoder = null; this.latest = null; this.frames = 0;
    }
    stop() {
      this.generation++; this.controller?.abort(); clearTimeout(this.reconnectTimer); clearInterval(this.watchdog);
      cancelAnimationFrame(this.paintRequest);
      if (this.decoder && this.decoder.state !== 'closed') this.decoder.close();
      if (this.latest) { this.latest.close(); this.latest = null; }
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    start(token, url = '/api/video', format = 'h264') {
      this.stop(); this.token = token; this.url = url; this.frames = 0;
      if (format === 'mjpeg') { this.connectMjpeg(this.generation); return; }
      if (!('VideoDecoder' in window) || !('EncodedVideoChunk' in window)) {
        this.callbacks.fallback?.('This browser does not support H.264 WebCodecs. Showing sampled JPEG preview.'); return;
      }
      this.connect(this.generation);
    }
    async connectMjpeg(generation) {
      if (generation !== this.generation) return;
      const controller = new AbortController(); this.controller = controller;
      let reader, received = 0, lastFrame = performance.now(), failure;
      this.callbacks.status?.('Connecting webcam');
      const watchdog = setInterval(() => {
        if (generation === this.generation && performance.now() - lastFrame > 7000) {
          this.stop(); this.callbacks.fallback?.('No webcam frames received. Showing sampled JPEG preview.');
        }
      }, 1000);
      this.watchdog = watchdog;
      try {
        const response = await fetch(this.url, {redirect:'error', headers:{Authorization:'Bearer ' + this.token}, cache:'no-store', signal:controller.signal});
        if (generation !== this.generation || controller.signal.aborted) { void response.body?.cancel().catch(() => {}); return; }
        if (!response.ok || !response.body) throw Error('Webcam stream unavailable (' + response.status + ')');
        const match = /^multipart\/x-mixed-replace;\s*boundary=([A-Za-z0-9_-]{1,70})$/.exec(response.headers.get('Content-Type') || '');
        if (!match) throw Error('Invalid webcam stream type');
        const parser = new PortalMjpegParser(match[1]); reader = response.body.getReader();
        while (!controller.signal.aborted && generation === this.generation) {
          const {value,done} = await reader.read(); if (done) break;
          const frames = parser.push(value);
          // Show only the newest complete frame in each network chunk.
          if (frames.length) {
            const frame = await createImageBitmap(new Blob([frames[frames.length-1]], {type:'image/jpeg'}));
            try {
              if (generation !== this.generation || controller.signal.aborted) return;
              if (frame.width > 4096 || frame.height > 4096) throw Error('Webcam dimensions exceed limit');
              this.canvas.width=frame.width; this.canvas.height=frame.height; this.context.drawImage(frame,0,0);
              received++; this.frames++; lastFrame=performance.now(); this.callbacks.frame?.();
            } finally { frame.close(); }
          }
        }
      } catch (error) { failure=error; }
      finally {
        clearInterval(watchdog); controller.abort(); if (reader) void reader.cancel().catch(() => {});
        if (generation === this.generation) {
          if (!received && !this.frames) this.callbacks.fallback?.((failure?.message || 'No webcam frames received') + '. Showing sampled JPEG preview.');
          else { this.callbacks.status?.('Reconnecting webcam'); this.reconnectTimer=setTimeout(() => this.connectMjpeg(generation),750); }
        }
      }
    }
    paint(generation) {
      if (generation !== this.generation) return;
      if (this.latest) {
        const frame = this.latest; this.latest = null;
        this.canvas.width = frame.displayWidth; this.canvas.height = frame.displayHeight;
        this.context.drawImage(frame, 0, 0); frame.close();
        this.callbacks.frame?.();
      }
      this.paintRequest = requestAnimationFrame(() => this.paint(generation));
    }
    async connect(generation) {
      if (generation !== this.generation) return;
      const controller = new AbortController(); this.controller = controller;
      let decoder, reader, watchdog, failure = null, received = 0;
      let lastFrame = performance.now();
      this.callbacks.status?.('Connecting video');
      try {
        decoder = new VideoDecoder({
          output: frame => {
            if (generation !== this.generation || controller.signal.aborted) { frame.close(); return; }
            if (this.latest) this.latest.close();
            this.latest = frame; this.frames++; received++; lastFrame = performance.now();
          },
          error: error => { failure = error; controller.abort(); },
        });
        this.decoder = decoder; decoder.configure({codec: 'avc1.42E028', optimizeForLatency: true});
        watchdog = setInterval(() => {
          if (generation === this.generation && performance.now() - lastFrame > 7000) {
            failure = Error('No video frames received');
            this.stop();
            this.callbacks.fallback?.(failure.message + '. Showing sampled JPEG preview.');
          }
        }, 1000);
        this.watchdog = watchdog;
        const response = await fetch(this.url || '/api/video', {redirect: 'error', headers: {Authorization: 'Bearer ' + this.token}, cache: 'no-store', signal: controller.signal});
        if (generation !== this.generation || controller.signal.aborted) { await response.body?.cancel().catch(() => {}); return; }
        if (!response.ok || !response.body) throw Error('Video stream unavailable (' + response.status + ')');
        reader = response.body.getReader(); let pending = new Uint8Array(0), waitKey = true;
        let startTimestamp = null, arrivalStart = 0, newestTimestamp = 0;
        cancelAnimationFrame(this.paintRequest); this.paint(generation);
        while (!controller.signal.aborted && generation === this.generation) {
          const {value, done} = await reader.read(); if (done) break;
          if (pending.length + value.length > MAX_PACKET_BYTES * 2 + 26) throw Error('Video buffer limit exceeded');
          const joined = new Uint8Array(pending.length + value.length); joined.set(pending); joined.set(value, pending.length); pending = joined;
          let offset = 0;
          while (pending.length - offset >= 13) {
            const header = new DataView(pending.buffer, pending.byteOffset + offset, 13), size = header.getUint32(0);
            if (size < 1 || size > MAX_PACKET_BYTES) throw Error('Invalid video packet size');
            if (pending.length - offset < 13 + size) break;
            const timestamp = Number(header.getBigUint64(4)), key = header.getUint8(12) !== 0;
            if (!Number.isSafeInteger(timestamp)) throw Error('Invalid video timestamp');
            if (startTimestamp === null) { startTimestamp = timestamp; arrivalStart = performance.now(); }
            newestTimestamp = timestamp;
            await waitCapacity(decoder, controller.signal);
            if (!waitKey || key) {
              waitKey = false;
              decoder.decode(new EncodedVideoChunk({type: key ? 'key' : 'delta', timestamp, data: pending.slice(offset + 13, offset + 13 + size)}));
            }
            offset += 13 + size;
          }
          pending = pending.slice(offset);
          if (startTimestamp !== null && performance.now() - arrivalStart - (newestTimestamp - startTimestamp) / 1000 > 1500) throw Error('Reconnecting to current video');
        }
      } catch (error) { if (!failure) failure = error; }
      finally {
        clearInterval(watchdog);
        if (generation === this.generation) cancelAnimationFrame(this.paintRequest);
        controller.abort();
        // Network cancellation must not hold decoder cleanup or preview fallback.
        if (reader) void reader.cancel().catch(() => {});
        if (decoder && decoder.state !== 'closed') decoder.close();
        if (generation === this.generation && this.latest) { this.latest.close(); this.latest = null; }
        if (generation === this.generation) {
          if (received === 0 && this.frames === 0) {
            this.callbacks.fallback?.((failure?.message || 'No video frames received') + '. Showing sampled JPEG preview.');
          } else {
            this.callbacks.status?.('Reconnecting video');
            this.reconnectTimer = setTimeout(() => this.connect(generation), 750);
          }
        }
      }
    }
  }
  class PortalMjpegParser {
    constructor(boundary) { this.boundary=boundary; this.pending=new Uint8Array(0); }
    push(value) {
      const limit=2*1024*1024;
      if (this.pending.length + value.length > limit + 8192) throw Error('Webcam buffer limit exceeded');
      const joined=new Uint8Array(this.pending.length+value.length); joined.set(this.pending); joined.set(value,this.pending.length); this.pending=joined;
      const frames=[];
      while (this.pending.length) {
        let end=-1;
        for (let i=0;i<Math.min(this.pending.length-3,4096);i++) {
          if (this.pending[i]===13 && this.pending[i+1]===10 && this.pending[i+2]===13 && this.pending[i+3]===10) {end=i;break;}
        }
        if (end<0) { if (this.pending.length>4096) throw Error('Webcam header limit exceeded'); break; }
        const header=new TextDecoder('ascii').decode(this.pending.slice(0,end));
        const lines=header.replace(/^\r\n/,'').split('\r\n');
        if (lines.shift()!=='--'+this.boundary) throw Error('Invalid webcam boundary');
        const fields={};
        for (const line of lines) { const split=line.indexOf(':'); if (split<1) throw Error('Invalid webcam header'); const key=line.slice(0,split).toLowerCase(); if (key in fields) throw Error('Duplicate webcam header'); fields[key]=line.slice(split+1).trim(); }
        if (fields['content-type']!=='image/jpeg') throw Error('Invalid webcam frame type');
        const size=Number(fields['content-length']);
        if (!/^\d+$/.test(fields['content-length'] || '') || !Number.isInteger(size) || size<1 || size>limit) throw Error('Invalid webcam frame size');
        if (this.pending.length<end+4+size+2) break;
        if (this.pending[end+4+size]!==13 || this.pending[end+5+size]!==10) throw Error('Invalid webcam frame terminator');
        frames.push(this.pending.slice(end+4,end+4+size)); this.pending=this.pending.slice(end+6+size);
      }
      return frames;
    }
  }
  window.PortalMjpegParser=PortalMjpegParser;
  window.PortalVideo = PortalVideo;
})();
