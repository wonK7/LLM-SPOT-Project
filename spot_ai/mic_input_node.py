import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
import numpy as np
import whisper
import subprocess


class MicInputNode(Node):
    def __init__(self):
        super().__init__('mic_input_node')
        self.publisher_ = self.create_publisher(String, '/spot_ai/voice_text', 10)
        self.status_publisher_ = self.create_publisher(String, '/spot_ai/status', 10)
        self.model = whisper.load_model('base')
        self.fs = 16000  # Sample rate
        self.declare_parameter('mic_device', -1)
        self.declare_parameter('use_pulse_fallback', True)
        self.declare_parameter('pulse_source', 'RDPSource')
        self.declare_parameter('whisper_language', 'auto')
        self.declare_parameter('record_duration_sec', 6.0)
        self.declare_parameter('record_interval_sec', 0.5)
        self.duration = max(0.5, float(self.get_parameter('record_duration_sec').value))
        self.record_interval_sec = max(0.0, float(self.get_parameter('record_interval_sec').value))
        self.input_device = self._resolve_input_device()
        self.use_pulse_fallback = bool(self.get_parameter('use_pulse_fallback').value)
        requested_pulse_source = str(self.get_parameter('pulse_source').value).strip()
        self.pulse_source = self._resolve_pulse_source(requested_pulse_source)
        self.whisper_language = str(self.get_parameter('whisper_language').value).strip()
        language_label = self.whisper_language if self.whisper_language else 'auto'
        self.get_logger().info(
            f'MicInputNode started. Speak only while recording. '
            f'Window={self.duration}s gap={self.record_interval_sec}s language={language_label}'
        )
        self.timer = self.create_timer(self.duration + self.record_interval_sec, self.capture_and_publish)

    def _resolve_input_device(self):
        requested = int(self.get_parameter('mic_device').value)
        try:
            devices = sd.query_devices()
        except Exception as exc:
            self.get_logger().error(f'Failed to query audio devices: {exc}')
            return None

        # If user specified a device index, validate and use it.
        if requested >= 0:
            try:
                info = sd.query_devices(requested)
                if info.get('max_input_channels', 0) > 0:
                    self.get_logger().info(f'Using mic_device={requested}: {info.get("name", "unknown")}')
                    return requested
                self.get_logger().error(f'mic_device={requested} is not an input device.')
            except Exception as exc:
                self.get_logger().error(f'Invalid mic_device={requested}: {exc}')

        # Otherwise pick first input-capable device.
        for idx, info in enumerate(devices):
            if info.get('max_input_channels', 0) > 0:
                self.get_logger().info(f'Using auto-selected input device {idx}: {info.get("name", "unknown")}')
                return idx

        self.get_logger().error('No input audio device found. Check microphone permissions/device mapping.')
        return None

    def _run_pactl(self, *args):
        try:
            return subprocess.run(
                ['pactl', *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as exc:
            self.get_logger().warning(f'Failed to query PulseAudio via pactl: {exc}')
            return None

    def _list_pulse_sources(self):
        result = self._run_pactl('list', 'short', 'sources')
        if result is None or result.returncode != 0:
            if result and result.stderr.strip():
                self.get_logger().warning(f'pactl list sources failed: {result.stderr.strip()}')
            return []

        sources = []
        for line in result.stdout.splitlines():
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1].strip():
                sources.append(parts[1].strip())
        return sources

    def _get_default_pulse_source(self):
        result = self._run_pactl('info')
        if result is None or result.returncode != 0:
            return ''

        for line in result.stdout.splitlines():
            if line.startswith('Default Source:'):
                return line.split(':', 1)[1].strip()
        return ''

    def _resolve_pulse_source(self, requested_source):
        sources = self._list_pulse_sources()
        if not sources:
            self.get_logger().warning('No PulseAudio sources reported by pactl.')
            return requested_source

        if requested_source:
            if requested_source in sources:
                self.get_logger().info(f'Using requested Pulse source: {requested_source}')
                return requested_source

            lowered = requested_source.lower()
            partial_matches = [source for source in sources if lowered in source.lower()]
            if partial_matches:
                self.get_logger().warning(
                    f"Requested Pulse source '{requested_source}' not found. "
                    f"Using partial match '{partial_matches[0]}' instead."
                )
                return partial_matches[0]

        default_source = self._get_default_pulse_source()
        if default_source:
            self.get_logger().warning(
                f"Requested Pulse source '{requested_source or 'unset'}' not found. "
                f"Falling back to default source '{default_source}'."
            )
            return default_source

        self.get_logger().warning(
            f"Requested Pulse source '{requested_source or 'unset'}' not found and no default source is available."
        )
        return requested_source

    def _capture_with_parec(self):
        cmd = ['parec']
        if self.pulse_source:
            cmd.extend(['--device', self.pulse_source])
        cmd.extend([
            '--rate', str(self.fs),
            '--channels', '1',
            '--format', 's16le',
            '--raw'
        ])
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.duration,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or b''
            stderr = (exc.stderr or b'').decode(errors='ignore').strip()
        except Exception as exc:
            self.get_logger().error(f'Pulse capture failed: {exc}')
            return None
        else:
            raw = result.stdout
            stderr = result.stderr.decode(errors='ignore').strip()
            if result.returncode not in (0,):
                self.get_logger().warning(
                    f"parec exited early with code {result.returncode}"
                    + (f": {stderr}" if stderr else '.')
                )

        if not raw:
            details = f" source={self.pulse_source}" if self.pulse_source else ''
            if stderr:
                self.get_logger().error(f'Pulse capture returned empty audio stream.{details} stderr={stderr}')
            else:
                self.get_logger().error(f'Pulse capture returned empty audio stream.{details}')
            return None

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def capture_and_publish(self):
        audio = None

        if self.input_device is not None:
            self.get_logger().info(
                f'Recording now for {self.duration} seconds from microphone. Speak one short command.'
            )
            try:
                audio = sd.rec(
                    int(self.duration * self.fs),
                    samplerate=self.fs,
                    channels=1,
                    dtype='float32',
                    device=self.input_device
                )
                sd.wait()
                audio = np.squeeze(audio)
            except Exception as exc:
                self.get_logger().error(f'Audio capture failed: {exc}')
                audio = None

        if audio is None and self.use_pulse_fallback:
            self.get_logger().info(
                f'Recording now for {self.duration} seconds from Pulse source: {self.pulse_source}. '
                'Speak one short command.'
            )
            audio = self._capture_with_parec()

        if audio is None:
            self.get_logger().error('Skipping recording: no available input device.')
            return

        self.get_logger().info('Recording complete. Do not speak now; transcribing...')
        transcribe_kwargs = {
            'fp16': False,
            'temperature': 0.0,
            'condition_on_previous_text': False,
        }
        if self.whisper_language and self.whisper_language.lower() != 'auto':
            transcribe_kwargs['language'] = self.whisper_language
        result = self.model.transcribe(audio, **transcribe_kwargs)
        text = result['text'].strip()
        if not text:
            self.get_logger().warning('Recognized text is empty.')
            return
        self.get_logger().info(f'Recognized: {text}')
        self.get_logger().info('Waiting for the next recording window...')
        msg = String()
        msg.data = text
        self.publisher_.publish(msg)
        status_msg = String()
        status_msg.data = f'Heard: {text}'
        self.status_publisher_.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MicInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
