"""Questions the bot's own lookups answer quickly and correctly, each passing in scripts/question_bank.py (2026-11-05). route_to_known_question maps what a person typed onto one of
these with their real values (a build, workflow or number from an example is rejected by _known_question_invents): one quick LLM call, then a tested lookup, instead of a 30 s investigation that works out its own queries and can differ from run to run."""

KNOWN_QUESTIONS = """\
Overall state
- what is failing in the latest IB
- what is failing in the latest IB on el9_amd64_gcc14
- status of 20_0
- what would you tell the next shifter in one sentence
- give me the morning report
- did anything get fixed today?
- is the IB getting better or worse this week?
- summary of CMSSW_20_1_ASAN_X_2026-11-02-2300
- how is the ASAN IB doing?
- which sub-IB is in the worst shape?
- which architecture has the most failures in the latest IB?
- which release series are active?
- which release series has the most failing RelVals right now?
- which IB this week had the most failed relvals?
- is el10 ready?
- did 16_0 aarch64 finish?
- which architectures does CMSSW_20_1_CPP23_X_2026-11-03-1100 have?
- which IB flavours exist for 20_1?
- latest 16_1 IB
- how fresh is your data?
Builds
- build errors in CMSSW_20_1_X_2026-11-01-2300
- which build types failed to build yesterday
- NONLTO warnings in the latest build?
Comparing
- compare 17_1 latest with previous on el8_amd64_gcc13
- compare CLANG yesterday with the day before
- is NONLTO worse than the primary in the newest IB?
- which workflows failed on both el8 and el9 in CMSSW_20_1_X_2026-11-03-1100?
RelVals
- relval failures in CMSSW_5_3_X
- which workflows fail on fc42_riscv64_gcc15?
- which workflows timed out in the latest IB?
- which workflows have exit code 25600 in CMSSW_20_1_X_2026-11-03-1100?
- which workflows have a segfault this week?
- which relval has been failing the longest?
- which relval fails on the most architectures?
- intermittent failures in relvals this week
- known errors list for aarch64 20_1
One workflow
- why is 10805.31 failing in 16_1?
- is 11834.0 failing anywhere?
- history of workflow 1370.18
- how often has 539.0 failed in the last week?
- which step fails for 1349.0
- show me the log of 140.5611
- has 537.0 been reported?
- suggest a fix for 539.0, give me a PR title and description
Unit tests
- list failing unit tests in the latest IB
- unit test failures on el9_aarch64_gcc14 in 20_1
- how many unit tests failed yesterday
- which package has the most unit test failures this month?
- unit tests of HLTrigger/Configuration
- why does TestIntegrationEventSetup fail on aarch64?
- addon tests for CMSSW_20_1_X_2026-11-03-1100
Errors and test logs
- what does exit code 134 mean
- what does SonicCallFailed mean?
- what's the most common exception this week?
- any hidden problems in passed tests in CMSSW_20_1_X_2026-10-12-2300?
- are there sanitizer errors in passed tests in CMSSW_20_1_ASAN_X_2026-11-02-2300?
- what is Bus error in 17_0 unit logs?
Clang
- clang warnings in the latest IB
- how many clang warnings does FWCore have?
- which package has the most clang warnings
- which file has the most clang warnings?
- are clang results up to date?
GitHub
- what does PR 52223 do
- who authored PR 52223
- issue 51793
- could PR 52223 have broken the CLANG IB?
- which PRs merged yesterday?
- which cmsdist PRs were merged in the last 30 days
- PRs by mmusich merged recently
- issues mentioning TrackstersProducer
- any open PR that mentions ROCm?
- how many open issues are there
- issues opened by mmusich
- how many PRs by tvami were merged in October 2026?
- how many PRs were merged on 2026-10-25?
- latest open issue
Meaning
- what is a RelVal
- what does NONLTO mean
"""
