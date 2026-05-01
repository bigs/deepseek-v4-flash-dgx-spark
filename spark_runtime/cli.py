"""Small CLI probe for the long-lived Spark engine."""

from __future__ import annotations

import argparse

from spark_runtime.engine import DeepSeekSparkEngine, config_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--cache-policy", choices=["reuse", "reset", "none"], default="reset")
    args = parser.parse_args()

    import asyncio

    engine = DeepSeekSparkEngine(config_from_env())
    result = asyncio.run(
        engine.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            session_id=args.session_id,
            cache_policy=args.cache_policy,
        )
    )
    print(result.generated_text)
    print(result)
    engine.close()


if __name__ == "__main__":
    main()
