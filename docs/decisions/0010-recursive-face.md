# ADR 0010 — PaperLab's face is Recursive

Status: proposed by Claude on the owner's delegation, 2026-10-28. Reversible.

## Context
ADR 0004 left the display face open (candidate: Bricolage Grotesque). Bricolage belongs to Slop Editorial.

## Decision
PaperLab uses Recursive, a variable face with a `wght` axis, so the sentence can change weight without changing width. JetBrains Mono stays for numbers.

## Status of the build
Recursive is not in this repo's lockfile and the sandbox is offline, so it could not be added in phase 3a. The room ships with the house Inter variable face, which also has a `wght` axis, and the type is wired to the `--room-font` token. Adding Recursive is a follow-up: add the woff2 under `public/fonts/`, set `--room-font`, and check the 92 KB font budget.
