# Experimental datasets

The source archives contain the main benchmark and generalization data. `archives.json` lists file sizes and SHA-256 hashes; `events.json` maps the experimental incidents to archive members.

## Main benchmark

Online Boutique, Sock Shop, and Train Ticket each contribute 90 incidents. `TORAI_Main_270.zip` is the author-published, preprocessed representation. The RE2 raw archives provide request-level observations needed by the supplementary experiments. These are different representations of the same benchmark incidents, not additional independent events.

The CARE comparison uses Online Boutique and Train Ticket from this benchmark. Sock Shop has no corresponding request-level trace input. Train Ticket metrics are read from the matching event in `TORAI_Main_270.zip`.

## Generalization

`RE3_OB_Generalization_30.zip` contains the separate 30-event Online Boutique generalization set. The event mapping preserves the association between the experiment's `case_...` identifiers and source event directories.

## Prepared inputs

`../data/prepared_observations.jsonl` contains metric fields, normal operation statistics, incident operation bins, and normal-role counts. Evaluation labels are separate. These observations support the main method, six configurations, observation windows, and generalization runs without parsing the source archives again.

The supplementary input asset contains CARE normal/incident feature files and compact normal-span role observations. Intermediate metric and operation evidence is retained because it is the input held fixed in those experiments. Published rankings and metric tables are not included.

## Attribution

- TORAI data: https://doi.org/10.6084/m9.figshare.31925976.v1
- RCAEval RE2/RE3 data: https://doi.org/10.5281/zenodo.14590730
- RCAEval repository: https://github.com/phamquiluan/RCAEval
- Train Ticket trace revision: `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e` in `phamquiluan/RCAEval`. File-level sources are in the archive's `FILE_MANIFEST`.

The source records distribute the datasets under CC BY 4.0. Preserve attribution to the original authors, source links, and the license: https://creativecommons.org/licenses/by/4.0/ . Derived feature inputs and compact role records are transformations of those observations; they are not new raw measurements.
