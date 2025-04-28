#!/usr/bin/env python3
import sys
import json
import struct
import os
from resume_evaluator import ResumeEvaluatorAgent

def get_message():
    """Read a message from stdin and decode it."""
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length:
        sys.exit(0)
    message_length = struct.unpack('=I', raw_length)[0]
    message = sys.stdin.buffer.read(message_length).decode('utf-8')
    return json.loads(message)

def send_message(message):
    """Send an encoded message to stdout."""
    encoded_message = json.dumps(message).encode('utf-8')
    sys.stdout.buffer.write(struct.pack('=I', len(encoded_message)))
    sys.stdout.buffer.write(encoded_message)
    sys.stdout.buffer.flush()

def main():
    evaluator = ResumeEvaluatorAgent()
    
    while True:
        try:
            message = get_message()
            if message.get('action') == 'evaluateResume':
                # Get the file path from the message
                file_path = message.get('filePath')
                if not file_path:
                    send_message({'error': 'No file path provided'})
                    continue
                
                # Process the resume
                result = evaluator.process(file_path)
                send_message(json.loads(result))
                
        except Exception as e:
            send_message({'error': str(e)})

if __name__ == '__main__':
    main() 