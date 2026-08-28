"""The quizzes for the banks scenario.

Quiz text names real banks and real measured numbers, so it belongs to a scenario
rather than to fedviz. Every answer below is checked by `checks/verify_banks.py` or
`checks/verify_seeds.py`; nothing here is asserted from intuition.
"""

QUIZZES = {
    "board_ask": (
        "The board has asked for one shared model. What have they not yet been told?",
        ["Which algorithm the consortium should use.",
         "Whether a shared model can match what pooling the records would give.",
         "How much the network traffic will cost.",
         "Which bank has the most customers."],
        1,
        "Nobody has established whether collaboration can match pooling, or what it is "
        "worth to each bank. That is what part 1 measures, and every later part depends "
        "on the answer."),

    "travels_worst": (
        "Before you look: whose local model will do worst on the other banks' customers?",
        ["A Prime Metro, because it has the most customers and will overfit them.",
         "B Digital Growth, because its customers are the youngest.",
         "D Community High-Touch, because it has seen the fewest defaults.",
         "They will all be about the same."],
        2,
        "D Community High-Touch, averaging 0.786 away from home against 0.806 at home. "
        "The reason is worth more than the answer: it has seen 52 defaults. Across ten "
        "draws, how well a bank recovers the shared rule correlates +0.98 with the number "
        "of defaults it has seen and only +0.18 with how many customers it holds. "
        "A Prime Metro holds eight times as many customers and travels almost as badly, "
        "because at a 6% default rate it has seen only 63."),

    "matched_work": (
        "At the same total amount of training work, do the two schemes end up in the "
        "same place?",
        ["Yes — the same work gives the same model.",
         "No. More local work between exchanges reaches a given loss with fewer "
         "exchanges but more computing.",
         "No, more local work is always worse.",
         "It depends only on the number of customers."],
        1,
        "Reaching the one per cent target takes 56 exchanges at one local step and 4 at "
        "twenty, but 56 local steps per bank against 80. The network bill falls and the "
        "computing bill rises. Neither meter is the whole picture, which is why part 4 "
        "plots both."),

}
