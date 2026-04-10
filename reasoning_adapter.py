def detect_type(query: str) -> str:
    if any(word in query.lower() for word in ["calculate", "sum", "solve"]):
        return "math"
    elif any(word in query.lower() for word in ["law", "legal"]):
        return "legal"
    return "general"


def math_solver(query):
    return "Using math reasoning..."


def legal_reasoner(query):
    return "Using legal reasoning..."


def general_llm(query):
    return "Using general LLM..."


def router(query):
    q_type = detect_type(query)

    if q_type == "math":
        return math_solver(query)
    elif q_type == "legal":
        return legal_reasoner(query)
    else:
        return general_llm(query)


# Example
if __name__ == "__main__":
    print(router("solve 2+2"))
