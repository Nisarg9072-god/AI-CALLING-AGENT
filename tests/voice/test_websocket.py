"""
Tests for WebSocket transport and API integration.
"""
import pytest
import asyncio
import base64
from app.voice.transport.websocket_transport import WebSocketTransport
from app.voice.audio import AudioFrame, INTERNAL_SAMPLE_RATE


@pytest.mark.asyncio
class TestWebSocketTransport:
    async def test_handle_audio_chunk(self):
        sent_messages = []
        async def mock_send(data):
            sent_messages.append(data)
            
        transport = WebSocketTransport(send_fn=mock_send)
        transport.start_session()
        
        # Create a dummy base64 audio chunk
        raw_pcm = b"\x01\x00" * 100
        b64_pcm = base64.b64encode(raw_pcm).decode("ascii")
        
        await transport.handle_message({
            "type": "audio_chunk",
            "data": b64_pcm,
            "sample_rate": 16000
        })
        
        assert len(transport._audio_chunks) == 1
        
        # Signal end of audio
        await transport.handle_message({"type": "audio_end"})
        
        # Get the utterance
        utterance = await transport.get_utterance(timeout=1.0)
        assert utterance is not None
        assert len(utterance.samples) == 200
        
    async def test_ping_pong(self):
        sent_messages = []
        async def mock_send(data):
            sent_messages.append(data)
            
        transport = WebSocketTransport(send_fn=mock_send)
        await transport.handle_message({"type": "ping"})
        
        assert len(sent_messages) == 1
        assert sent_messages[0]["type"] == "pong"
        
    async def test_send_audio_frame(self):
        sent_messages = []
        async def mock_send(data):
            sent_messages.append(data)
            
        transport = WebSocketTransport(send_fn=mock_send)
        
        frame = AudioFrame(samples=b"\x05\x00\x06\x00")
        await transport.send_audio_frame(frame)
        
        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert msg["type"] == "audio"
        assert msg["format"] == "pcm16"
        assert base64.b64decode(msg["data"]) == b"\x05\x00\x06\x00"
