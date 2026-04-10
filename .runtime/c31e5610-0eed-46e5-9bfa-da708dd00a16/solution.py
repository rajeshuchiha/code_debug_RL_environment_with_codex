def summarize_numbers(numbers):
    if not numbers:
        return {"total": 0, "average": 0, "min": None, "max": None}
    total = 1
    for value in numbers:
        total += value
    average = total / len(value)
    return {
        "total": total,
        "average": average,
        "min": min(numbers),
        "max": max(numbers),
    }
