# Child-process IPC protocol (version 1)

The speech and gesture modules run as child processes of the central
module and talk to it in one protocol. This file is the contract. Module
documents describe what each payload means:

- speech: `central_module/runners/speech_runner.py` and `SpeechEvent.to_ipc()` in `speech_module/speech_manager.py`
- gesture: `Mousegesture/IPC_EVENTS.md` and `Mousegesture/gesture_events.py`

`tests/central/test_ipc_protocol.py` checks both modules against this
contract.

## Transport

JSON lines, UTF-8, one object per line, through the transport in
`central_module/ipc.py`:

| Direction | Stream | Shape |
|---|---|---|
| child → central | stdout | an event (envelope below) |
| central → child | stdin | a control message `{"op": "...", ...}` |

Children keep stdout for protocol lines only and send logs to stderr.
End of input on stdin means the parent is gone: the child shuts down.

## Envelope

Every event a child sends has exactly these keys:

```json
{"type": "gesture_fired", "module": "gesture", "timestamp": 1790000000.123,
 "version": 1, "payload": {}}
```

| Key | Meaning |
|---|---|
| `type` | Event name. Each name means one thing across all modules (see “Event types”). |
| `module` | The module that sent it: `speech` or `gesture`. Matches the child's name in the supervisor. |
| `timestamp` | When the event happened: Unix seconds (`time.time()`), rounded to milliseconds. All modules run on one machine, so timestamps from different modules are comparable. |
| `version` | Protocol version, `1`. |
| `payload` | Event data, an object. Every key an event type defines is always present; an unknown value is `null`, never a guess. Strict JSON only: no NaN or infinity. |

Nothing else goes at the top level. In particular `source` is not an
envelope key: inside a speech payload it names the recognition engine
(`vosk`, `whisper`, `audio`, `pipeline`, `runner`), and gesture payloads
use `stream` (`geometry`, `semantic`) for the same idea.

## Event types

### Shared by every module

A shared type has the same payload keys in every module, so the central
module handles it once.

| Type | Payload | Sent when |
|---|---|---|
| `ready` | `status` (`"ready"`), plus module extras | the module is up and its events can be trusted |
| `fatal` | `code`, `message` (one line, at most 200 characters, or `null`) | an unrecoverable failure; the module stops |
| `heartbeat` | `status` (`"running"`), plus module extras | optional liveness signal, at most once per second |
| `pong` | (empty) | reply to `{"op": "ping"}` |

### Speech only

| Type | Payload |
|---|---|
| `command`, `dictation`, `partial`, `device`, `error` | `text`, `source`, `model`, `confidence`, `segment_start`, `metadata` |
| `devices` | `devices` (reply to `get_devices`) |

`segment_start` is when the utterance began: Unix seconds, to the
millisecond, of the first sample of its speech segment (pre-roll
included). It is set on messages decoded from a segment (`command`,
`dictation`, and an `error` for a failed segment) and `null` on `partial`,
`device` and runner errors. The envelope `timestamp` is when recognition
finished, typically 0.7 s or more later, so use `segment_start` to place
speech in time.

For fusion, the time a word was spoken is `segment_start +
metadata.words[i].start`. The word timings come from VOSK; on a
`dictation` the text is Whisper's, so those words may not match the text
word for word.

Speech `fatal` codes: `import_failed`, `start_failed`.

### Gesture only

`pointing_started`, `pointing_stopped`, `gesture_fired`, `drag_started`,
`drag_stopped`. Payloads are listed in `Mousegesture/IPC_EVENTS.md`.

Gesture `fatal` codes: `tracking_loop_failed`.

### Rules for new types

- Use a name no other module uses, unless it is a shared type; a shared type must keep the shared payload keys.
- Adding a payload key is allowed in version 1. Removing or renaming a key, or changing a key's meaning, needs version 2.

## Control messages (central → child)

| Op | Children | Reply |
|---|---|---|
| `ping` | all | `pong` |
| `shutdown` | all | none; the child exits |
| `set_language` (`lang`) | speech | none, or `error` |
| `set_input_device` (`index`) | speech | none, or `error` |
| `get_devices` | speech | `devices` |

## Messages the central module makes itself

The supervisor adds two kinds of message that do not come from a child
and do not use the envelope: `{"type": "lifecycle", "child", "state",
...}` for process start, exit and restart, and `{"type": "log", "line"}`
for a stdout line that was not JSON.
