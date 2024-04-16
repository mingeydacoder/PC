answer_01 =\
"""
-- BEGIN SOLUTION
SELECT "candidates" as id,
       COUNT(*) AS number_of_rows
  FROM candidates
union
SELECT "districts" as county,
       COUNT(*) AS number_of_rows
  FROM districts
union
SELECT "parties" as id,
       COUNT(*) AS number_of_rows
  FROM parties
-- END SOLUTION
"""
answer_02 =\
"""
-- BEGIN SOLUTION
select "earliset" as id, min(vote_tallied_at)
from districts
union
select "latest" as id, max(vote_tallied_at)
from districts

-- END SOLUTION

"""
answer_03 =\
"""
-- BEGIN SOLUTION
select county, town, polling_place, min(vote_tallied_at)
from districts
union
select county,town, polling_place, max(vote_tallied_at)
from districts
-- END SOLUTION

"""
answer_04 =\
"""
-- BEGIN SOLUTION
with cte as (
select "maximum" as id, sum(votes)
from regional_legislators
group by candidate_id
order by sum(votes) desc
limit 1),cte2 as (
select "minimum" as id, sum(votes)
from regional_legislators
group by candidate_id
order by sum(votes)
limit 1)

select *
from cte
union
select *
from cte2
-- END SOLUTION

"""
answer_05 =\
"""
-- BEGIN SOLUTION
with cte as (
select legislator_region, parties.name, candidates.name, sum(votes)
from regional_legislators
join candidates
on candidates.id = regional_legislators.candidate_id
join parties
on parties.id = candidates.party_id
group by candidate_id
order by sum(votes) desc
limit 1)


,cte2 as (
select legislator_region, parties.name, candidates.name, sum(votes)
from regional_legislators
join candidates
on candidates.id = regional_legislators.candidate_id
join parties
on parties.id = candidates.party_id
group by candidate_id
order by sum(votes) 
limit 1)

select *
from cte
union 
select *
from cte2
-- END SOLUTION

"""
answer_06 =\
"""
-- BEGIN SOLUTION
with cte as (
select legislator_region, parties.name, candidates.name, sum(votes)
from regional_legislators
join candidates
on candidates.id = regional_legislators.candidate_id
join parties
on parties.id = candidates.party_id
group by candidate_id
order by sum(votes) desc
limit 15)

select *
from cte
-- END SOLUTION

"""
answer_07 =\
"""
-- BEGIN SOLUTION
with cte as ( 
select name,sum(votes)
from party_legislators
join parties
on parties.id=party_legislators.party_id
group by party_id
order by sum(votes) desc
limit 13 offset 3)

select *
from cte
-- END SOLUTION
"""
answer_08 =\
"""
-- BEGIN SOLUTION
select county,name,sum(votes)
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where candidate_id=330
group by county
union
select county,name,sum(votes)
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where candidate_id=331
group by county
union
select county,name,sum(votes)
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where candidate_id=329
group by county

-- END SOLUTION
"""

answer_09 =\
"""
-- BEGIN SOLUTION
with cte as(
select county,name,sum(votes) as sum
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where candidate_id=330
group by county
union
select county,name,sum(votes)
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where candidate_id=331
group by county
union
select county,name,sum(votes)
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where candidate_id=329
group by county
)

select county, name, max(sum)
from cte
group by county
-- END SOLUTION
"""

answer_10 =\
"""
-- BEGIN SOLUTION
with cte as(
select county,town,name,sum(votes) as sum
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where county = "臺北市" and candidate_id=330
group by town
union
select county,town,name,sum(votes) as sum
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where county = "臺北市" and candidate_id=331
group by town
union
select county,town,name,sum(votes) as sum
from presidents
join districts
on districts.id = presidents.district_id
join candidates
on candidates.id = presidents.candidate_id
where county = "臺北市" and candidate_id=329
group by town)

select county, town, name, max(sum)
from cte
group by town
-- END SOLUTION
"""