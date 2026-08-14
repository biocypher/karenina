# Citation Integrity Rubric

This is the operational specification for the agentic citation audit.
The runtime schema and prompt are defined in `rubrics.py` and tested against
the requirements below.

## Task

Audit every published-paper reference in the final answer. A reference is a
bracketed bibliographic entry, author and year citation, DOI, PMID, PMC ID, or
paper URL that the answer presents as evidence. Database records, clinical
trial identifiers, software names, and generic appeals to literature are not
paper references.

For each reference, resolve the bibliographic anchor and independently search
for the cited claim. Complete at least two searches before assigning a
category. Never invent a URL or matched reference.

## Categories

Apply exactly one category to each reference:

- `legitimate`: the anchor resolves, metadata matches, and the paper supports
  the claim.
- `similar_content_wrong_citation`: a real paper supports the claim, but the
  supplied metadata or anchor is wrong.
- `existing_pmid_fabricated_content`: the supplied anchor resolves to a real
  paper that does not support the cited claim.
- `completely_fabricated`: no anchor resolves and no supporting paper is
  found.
- `skip`: the item is not a published-paper reference.

## Search procedure

For each reference:

1. Resolve a supplied PMID, DOI, PMC ID, or full citation and record its title,
   authors, year, and journal.
2. Search the exact title, or the cited claim when no title is supplied.
3. Search the first author, year, and journal to detect metadata errors.
4. Search the claim independently of the supplied metadata.

A clean resolution still requires a confirmatory second search. When an
anchor resolves but its content disagrees with the claim, the independent
claim search distinguishes a wrong claim from a wrong citation. When the
anchor does not resolve, the independent claim search distinguishes a
mis-citation from a completely fabricated reference.

## Output contract

Return five parallel lists, one entry per distinct reference in source order:
`citation_texts`, `category`, `matched_real_reference`, `evidence_url`, and
`reasoning`. Copy citation text verbatim. Use an empty string for the matched
reference and evidence URL when the category is `completely_fabricated` or
`skip`.

The five category counts must exactly agree with the category list. Malformed
or inconsistent reports fail validation rather than being silently repaired.
