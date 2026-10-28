# afk — step away; your agents don't

A small CLI on `ai.harnesses` for one everyday situation: you have been working
with `claude`, `codex` or `pi` in a directory, and you need to leave. `afk`
lists the conversations in that directory across every harness, opens one in the
harness's own TUI in a sandbox, lets you detach and close the laptop,
pushes work to run unattended, and comes back later to attach, watch, bring
home, or stop. Nothing has to be started "the afk way": it reads the
harnesses' own stores, so a conversation you began by typing `claude` an hour
ago is listed like any other.

```
$ afk
~/proj  ·  claude, codex, pi

  here
  a3f5  claude  migrate the billing tests to pytest     active 2m
  b81c  claude  add rate limiting to the webhook        idle 6h
  01a0  codex   audit the auth middleware               idle 3h

  remote
  c2d1  claude  billing · migrate the billing tests…  running   $0.031 ($0.085/h)   ← from a3f5 · amber-fox
  9e07  claude  add rate limiting to the webhook      finished  $0.012 ($0.085/h)   ← new, was running · ← from b81c · teal-owl
  4b2a  codex   audit the auth middleware             expired   $0.054              snapshot 1.1 GB, kept to Nov 18 · ← from 01a0 · red-elk
  /h: memory, every hour it runs; CPU adds while it computes

afk push <id> [--bg] · attach <id> · peek <id> · pull <id> [--files] · stop <id> · rm <id>
```

On a terminal that list is a picker, drawn under your prompt rather than
over the whole screen: arrow keys choose a conversation, Enter or a letter
runs the command shown for it as if you had typed it (`stop` asks first),
and `q` leaves the list in your scrollback. Piped, `afk` prints the list.

A key means the same thing on every row it appears on:

| key | |
|---|---|
| Enter | into the conversation: one here resumes in its harness's own TUI (`claude --resume <id>`, `codex resume <id>` or `pi --session <id>`), a pushed one is attached. One here that is open in another terminal is taken over, after a y: that process ends first, then the TUI opens here (`afk stop <id> && claude --resume <id>`). Where afk cannot end what drives it, Enter peeks |
| `p` | peek |
| `s` | stop: a pushed one's sandbox (its files are kept), or the process a conversation here is open in |
| `r` | rm: delete a pushed one's sandbox and snapshot |
| `u`, `b` | push, push `--bg` (a conversation here) |
| `l`, `f` | pull, pull `--files` (a pushed one) |

On a stopped or expired one, Enter resumes it. `stop` and `rm` ask first.

A conversation here that is open in another terminal is `in use`. afk sees
any `claude`, but a `codex` only when it was started as `codex resume <id>`:
a fresh `codex` names no conversation afk can find. A `pi` started in a
terminal is never seen: it shows no conversation in its process name.

## The verbs

| | |
|---|---|
| `afk [--all]` | this directory's conversations, and the ones pushed elsewhere, with a status; `--all` lists every one, not just the recent 15 |
| `afk push [id] [--as NAME] [--bg [--prompt TEXT]] [--hours H] [--git-history] [--no-network]` | copy it and this directory into a sandbox of its own and open the TUI there, or run it unattended with `--bg`. `--prompt` tells the unattended agent what to do instead of "continue from where it left off". The sandbox runs up to `H` hours (default 5); in a git repository the work goes on a branch, and `--git-history` gives the sandbox the repository's history too. The network is open; `--no-network` leaves only the model gateway |
| `afk attach <id>` | your terminal, back on its TUI; if you quit the TUI, attach opens it again in the sandbox. A stopped or expired one is resumed from its snapshot first. One pushed with `--bg` opens in the TUI there: its unattended agent is ended first, after a y if it is still mid-turn |
| `afk peek <id>` | follow a conversation's transcript like `tail -f`, read-only: a pushed one, or one here open in another terminal. A status line says it is following; it ends when the agent finishes its turn or stops running, or, for one a person drives, after 30s with nothing new. Ctrl-C stops watching, never the agent. A stopped one is resumed to read it and stopped again after |
| `afk pull <id> [--files]` | bring a conversation home and open it in the TUI here; `--files` also brings the work: in a git repository as a branch, otherwise the files it changed, after showing them and asking |
| `afk wait [id] [--every S]` | poll until it (or every pushed one that is running) finishes, stops, expires or goes; then ring the terminal bell and say what happened. Ctrl-C stops waiting, never the agent |
| `afk stop <id>` | stop its sandbox, keeping its files as a snapshot (`attach` resumes it; it is billed as storage until `rm` or its expiry, 30 days); for a conversation here, end the process it is open in and wait until it has exited (that terminal gets its prompt back; the conversation stays) |
| `afk rm <id>` | delete its sandbox and snapshot: nothing is kept, nothing more is billed |
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

A sandbox afk creates runs up to five hours — the platform's own default is
five *minutes*, which is shorter than any time away — and stops on its own
after that; `--hours` sets it (Hobby plans allow up to 0.75). Stopping loses
nothing: the sandbox is kept with a snapshot of its files, listed as
`expired`, and `afk attach` resumes it, conversation and files intact.

Push *copies*: the local conversation stays. Keep typing here and the list
shows both, the pushed one marked `← from a3f5`.

**Let it work while you're gone.**

```
$ afk push --bg --as tests --prompt "make the failing tests pass"
$ afk wait tests               # or come back later: `afk` marks what changed
$ afk peek tests               # tail what it is doing
$ afk pull tests --files       # bring the result home, its edits too, in the TUI
```

Unattended means no approval gate, so it never blocks. Without `--prompt`
it is asked to continue from where the conversation left off — fine when you
had just asked for something, useless when the last thing said was "not
yet".

To step in, `afk attach tests` opens it in the TUI in its sandbox, on the
same conversation. The unattended agent still holds that conversation after
its turn, so attach ends it first; mid-turn, it asks, since that ends the
turn. From then on it is a TUI like any other.

**Bring the work home.** `pull` alone brings only the conversation: your
files are never touched. `pull --files` also brings what the agent did.

In a git repository (pushed from its top), it comes home as a branch. Push
records your working tree as it was — uncommitted and untracked files
included — as a commit on top of HEAD, without touching your branch, index
or files, and the sandbox works on `afk/<label>` from that tree. Pull
commits whatever the agent left uncommitted and makes its commits your
branch `afk/<label>`. Your checkout is untouched; look at it and merge it
like any branch, and resolve conflicts with git:

```
$ afk pull tests --files
afk: afk/tests has 2 commits on the tree you pushed · `git log afk/tests` to look, `git merge afk/tests` to take it
$ git merge afk/tests
```

Without `--git-history` the sandbox sees one commit, your tree; with it, the
whole history, for tasks that read `git log` or `blame`.

Anywhere else, at push time afk records a hash of every file it copied,
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
| `running` | its TUI runs in the sandbox with no terminal on it (`afk attach` to open it), or an unattended agent is mid-turn (`afk peek` to watch, `afk attach` to step in) |
| `finished` | an unattended agent has finished its turn; `afk attach` opens it in the TUI there, `afk pull --files` brings it home |
| `open elsewhere` | a terminal is on it right now, maybe yours in another tab; `afk attach` takes it over |
| `idle` | its TUI has exited; `afk attach` reopens it there, `afk pull` brings it home |
| `stopped` | `afk stop` stopped its sandbox; its files are kept as a snapshot. `afk attach` resumes it, `afk rm` deletes it |
| `expired` | its sandbox ran out of time and stopped; the same as `stopped` |
| `gone` | its sandbox no longer exists; afk forgets it |
| `unreachable` | afk could not ask its sandbox right now (network, credentials); the record is kept |

`afk` marks a row `← new, was <status>` the first time it shows a changed
status. There is no notification service: run `afk` when you are back, or
`afk wait` to be told.

The SDK keeps no registry and never will: app state is the app's.

## What it costs

Each pushed conversation's row shows what its sandbox has cost so far, from
what the platform measured: active CPU (time actually computing; waiting on
the model does not count), memory by the minute it ran, data it sent,
and, once stopped, its snapshot's storage — at the region's rates from
[Vercel's pricing](https://vercel.com/docs/vercel-sandbox/pricing). A
running sandbox also shows what it costs per hour, `$0.031 ($0.085/h)`:
its memory, billed every hour it is up, busy or not. CPU adds while it
computes; the platform reports CPU and data sent when it stops, and the
total then includes them. `stop` and `rm` say what it cost. These are list prices: a
Hobby plan's allowance or a Pro plan's monthly credit is not subtracted.

The agent in the sandbox reaches its model through AI Gateway, which bills
per token — not through your claude or ChatGPT subscription.

## Network

The sandbox can reach anything: install packages, clone, call APIs. The
model key is never in the VM: the sandbox's firewall adds it to requests to
the gateway. `push --no-network` locks it down to the gateway (and npm, so
the CLI can be installed).

## Install

```bash
uv tool install "git+https://github.com/vercel-labs/ai-python@harnesses-experimental#subdirectory=examples/apps/afk"
```

Or from a checkout: `uv tool install ./examples/apps/afk`. Then run `afk`
in any directory where you use `claude`, `codex` or `pi`. afk keeps its own
files in `~/.afk/` and writes into your project only on `afk pull --files`.

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
