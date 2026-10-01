# afk — step away; your agents don't

A small CLI on `ai.harnesses` for one everyday situation: you have been working
with `claude` or `codex` in a directory, and you need to leave. `afk` lists
the conversations in that directory across both harnesses, opens one in the
harness's own TUI in a sandbox, lets you detach and close the laptop,
pushes work to run unattended, and comes back later to attach, watch, bring
home, or stop. Nothing has to be started "the afk way": it reads the
harnesses' own stores, so a conversation you began by typing `claude` an hour
ago is listed like any other.

```
$ afk
~/proj  ·  claude, codex

  here
  a3f5  claude  migrate the billing tests to pytest     active 2m
  b81c  claude  add rate limiting to the webhook        idle 6h
  01a0  codex   audit the auth middleware               idle 3h

  remote
  c2d1  claude  billing         running        ← from a3f5 · amber-fox

afk push <id> [--bg] · attach <id> · peek <id> · pull <id> [--files] · stop <id>
```

On a terminal that list is a picker, drawn under your prompt rather than
over the whole screen: arrow keys choose a conversation, Enter or a letter
runs the command shown for it as if you had typed it (`stop` asks first),
and `q` leaves the list in your scrollback. Piped, `afk` prints the list.

## The verbs

| | |
|---|---|
| `afk [--all]` | this directory's conversations, and the ones pushed elsewhere, with a status; `--all` lists every one, not just the recent 15 |
| `afk push [id] [--as NAME] [--bg] [--hours H]` | copy it and this directory into a sandbox of its own and open the TUI there, or run it unattended with `--bg`; the sandbox lives up to `H` hours (default 5) |
| `afk attach <id>` | your terminal, back on its TUI; if you quit the TUI, attach opens it again in the sandbox |
| `afk peek <id>` | watch an unattended agent's transcript, read-only |
| `afk pull <id> [--files]` | bring a conversation home and open it in the TUI here; `--files` also brings the files it changed there, after showing them and asking |
| `afk stop <id>` | end its sandbox |
| `afk setup` | choose the Vercel team afk's sandboxes use |

**Ctrl-]** detaches from a sandbox TUI and leaves it running. A pulled
conversation opens in an ordinary local TUI: quit it as you always do.

An id is whatever the list showed: a label you gave with `--as`, a word from a
title, or a short id prefix. With no id, `afk` chooses only when exactly one
conversation is clearly in use; with several in play it shows them and asks.
It never guesses.

## Three flows

**You were just working, normally.** In another tab:

```
$ afk push --as billing        # copies the conversation into a sandbox, opens its TUI
                               # work in it; Ctrl-] and close the laptop
$ afk attach billing           # next day, any machine: same TUI, scrollback intact
```

A sandbox afk creates lives up to five hours — the platform's own default is
five *minutes*, which is shorter than any time away — and stops on its own
after that; `--hours` sets it (Hobby plans allow up to 0.75).

Push *copies*: the local conversation stays. Keep typing here and the list
shows both, the pushed one marked `← from a3f5`.

**Let it work while you're gone.**

```
$ afk push --bg --as tests     # unattended, no approval gate, so it never blocks
$ afk peek tests               # tail what it is doing
$ afk pull tests --files       # bring the result home, its edits too, in the TUI
```

An unattended agent can be watched, pulled, or stopped, not taken over:
that is the SDK's live-reattach follow-up.

**Bring the work home.** `pull` alone brings only the conversation: your
files are never touched. `pull --files` also brings what the agent changed
in the sandbox. At push time afk records a hash of every file it copied,
so on pull it can tell the sandbox's changes from yours:

```
$ afk pull tests --files
afk: the sandbox changed 2 files:
       update  src/billing.py
       add     tests/test_billing.py
afk: 1 file changed here too since the push; writing would lose your local version:
       update  README.md
afk: write the 2 changes here? [y/N] y
afk: overwrite your version of 1 file? [y/N] n
afk: wrote 2 files; left 1 as they are
```

Nothing is written without a yes, and files you changed locally need a
second, separate yes. A sandbox pushed before afk recorded what it copied
has no baseline: every file that differs is listed as a possible conflict,
and nothing is deleted.

## Where things live, and why

`afk` keeps one file, `~/.afk/state.json`, written on push: the directory the
conversation came from, the label, the sandbox name, the pty name, the
SDK `Handle`, and a hash of each file the push copied. That is exactly the set of facts that exist only because *you*
did something; everything else is asked of the machine each time — the
harness lists its conversations and whether they are held, the workspace
lists its live ptys, a sandbox that no longer exists is shown once as `gone`
and forgotten.

Every push gets a sandbox of its own, with your directory copied as it is
at that moment. Two conversations pushed from one directory never share a
tree over there, so neither lands in files the other left behind, and
`stop` ends exactly one conversation. Each comes home on its own, with
`pull --files`.

A remote conversation's status says what you can do with it:

| status | meaning |
|---|---|
| `running` | its TUI runs in the sandbox with no terminal on it (`afk attach` to open it), or an unattended agent is mid-turn (`afk peek` to watch) |
| `finished` | an unattended agent has finished its turn; `afk pull --files` brings it home |
| `open elsewhere` | a terminal is on it right now, maybe yours in another tab; `afk attach` takes it over |
| `idle` | its TUI has exited; `afk attach` reopens it there, `afk pull` brings it home |
| `gone` | its sandbox no longer exists; afk forgets it |
| `unreachable` | afk could not ask its sandbox right now (network, credentials); the record is kept |

The SDK keeps no registry and never will: app state is the app's.

## Install

```bash
uv tool install "git+https://github.com/vercel-labs/ai-python@harnesses-experimental#subdirectory=examples/apps/afk"
```

Or from a checkout: `uv tool install ./examples/apps/afk`. Then run `afk`
in any directory where you use `claude` or `codex`. afk keeps its own files
in `~/.afk/` and writes into your project only on `afk pull --files`.

## Vercel, set up on first use

The first time afk needs a sandbox, it sets itself up. It asks only when it
has to:

1. It signs in through the Vercel CLI. Not installed: afk says how to install
   it (`npm i -g vercel`). Not logged in: afk runs `vercel login` for you.
2. It picks a team. With one team there is no question. With several, it
   lists them with your Vercel CLI's current team as the default: press
   Enter to accept. `AFK_TEAM=<slug>` decides without asking.
3. It creates a project named `afk-<random>` in that team for its sandboxes
   and records the ids in `~/.afk/config.json`. The file holds no secrets.

After that, every run mints a short-lived project token from your login, so
there is nothing to pull or refresh. `afk setup` chooses a team again.

The sandbox reaches a model through AI Gateway with that same token, so
`AI_GATEWAY_API_KEY` is optional. The first push checks, once, that the
team's AI Gateway will serve it; a team without a card on file is refused,
and afk says so and offers another team. With `AI_GATEWAY_API_KEY` set, afk
uses that key instead. Either way the key never enters the VM: the sandbox
injects it at egress.

Credentials already in the environment always win: `VERCEL_OIDC_TOKEN`, or
`VERCEL_TOKEN` with `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID`. On this
machine, the CLIs keep using your own login unless you set
`AI_GATEWAY_API_KEY`.

## Developing

```
cd examples/apps/afk
uv run afk                      # or: uv run python -m afk
```

Tests, in `examples/apps/afk`: `uv run pytest -m "not live"` runs the pure
parts offline; `uv run pytest` also runs the live listing tests, which need a
real `claude` and Vercel credentials.
