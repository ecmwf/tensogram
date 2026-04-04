# Features decided to implement

- API
	- [ ] Streaming API that allows encoder to progressively encode and transmit immediately on a socket without need for large buffers.

- CLI tools:
    - [ ] tool to merge messages (in same file or different files) to single message. they should take their free-form metadata, common and specific with them.
    - [ ] tool to split data objects from a single messages into multiple messages. they should take their free-form metadata, common and specific with them.
    - [ ] tool to reshuffle frames inside a message (move footer frames to header frames)

- Metadata:
	- [ ] implement inside metadata frame, the CBOR contents required: 'common', 'payload' and 'reserved'.

- Examples:
    - [ ] show-case the Streaming API

- Tools
	- [ ] implement converter to and from GRIB
	- [ ] implement split / merge messages
	- [ ] implement reshuffle message (move footer frames to header frames)

- Documentation:
	- [ ] shorten the landing README.md, moving to other linked documents information that may be too detailed.

- Builds
	- [ ] Review all the dependencies:
		- [ ] remove support for md5, sha hashes -- just keep xxh3
		- [ ] perform simplification of depedencies even if it means re-implementing some code inside tensogram, but ask user before removing a dependency while providing an explanation of why it is present and what it does.


# Code quality improvements (from code review)
All items completed — see DONE.md for details.
