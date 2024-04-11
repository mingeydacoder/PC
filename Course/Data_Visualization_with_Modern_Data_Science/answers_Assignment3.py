answer_01 =\
"""
-- BEGIN SOLUTION
SELECT name
  FROM sqlite_master
 WHERE name NOT LIKE 'sqlite%';
-- END SOLUTION
"""

answer_02 =\
"""
-- BEGIN SOLUTION
SELECT election_type_id,COUNT(*) AS number_of_candidates
FROM candidates
group by election_type_id;
-- END SOLUTION
"""

answer_03 =\
"""
SELECT county,
       COUNT(*) AS number_of_rows
  FROM districts
 GROUP BY county
 ORDER BY number_of_rows DESC;
"""

answer_04 =\
"""
-- BEGIN SOLUTION
SELECT county,
       COUNT(*) AS number_of_rows
  FROM districts
 GROUP BY county
 ORDER BY number_of_rows DESC
 limit 7;
-- END SOLUTION
"""

answer_05 =\
"""
-- BEGIN SOLUTION
select parties.name, count(*) 
from party_legislators
join candidates
on party_legislators.id=candidates.id
JOIN parties
ON candidates.party_id = parties.id
where candidates.election_type_id=2
group by candidates.party_id
order by count(*) desc
-- END SOLUTION
"""

answer_06 =\
"""
-- BEGIN SOLUTION
select parties.name, count(*) 
from party_legislators
join candidates
on party_legislators.id=candidates.id
JOIN parties
ON candidates.party_id = parties.id
where candidates.election_type_id=2
group by candidates.party_id
order by count(*) desc
limit 11;
-- END SOLUTION
"""

answer_07 =\
"""
SELECT county,legislator_region,number,parties.name,candidates.name,SUM(votes) AS sum_votes
from regional_legislators
join districts
on regional_legislators.district_id = districts.id
join candidates
on candidates.id = regional_legislators.candidate_id
join parties
on parties.id = candidates.party_id
where county = '臺北市'
group by candidate_id
order by legislator_region

"""

answer_08 =\
"""
with cte as(
SELECT county,legislator_region,number,parties.name,candidates.name,
SUM(votes) AS sum_votes
from regional_legislators
join districts
on regional_legislators.district_id = districts.id
join candidates
on candidates.id = regional_legislators.candidate_id
join parties
on parties.id = candidates.party_id
where county = '臺北市'
group by candidate_id
order by sum_votes desc
)
select *
from cte
group by legislator_region
"""

answer_09 =\
"""
with cte as (
select name,round(sum(votes)/13776736.0,4) as per
from party_legislators
join parties
on party_legislators.party_id = parties.id
group by name
order by per desc
),cte2 as (
select name,round(sum(votes)/13776736.0,4) as per
from party_legislators
join parties
on party_legislators.party_id = parties.id
group by name
order by per desc
limit 3
)

select *
from cte2
union
select '其他' as name,sum(per)
from cte
where per<0.2207
order by per desc
"""

answer_10 =\
"""
SELECT '不分區立委' AS election_type,
       parties.name AS party_name,
       round(SUM(party_legislators.votes)/(select sum(votes)*1.0 from party_legislators),4) AS votes_percentage
  FROM party_legislators
  JOIN parties
    ON party_legislators.party_id = parties.id
 WHERE parties.name IN ('中國國民黨', '台灣民眾黨', '民主進步黨')
 GROUP BY parties.name
 union
 SELECT '區域立委' AS election_type,
       parties.name AS party_name,
       round(SUM(regional_legislators.votes)/(select sum(votes)*1.0 from regional_legislators),4) AS votes_percentage
  FROM regional_legislators
  join candidates
  on regional_legislators.candidate_id = candidates.id
  JOIN parties
    ON candidates.party_id = parties.id
 WHERE parties.name IN ('中國國民黨', '台灣民眾黨', '民主進步黨')
 GROUP BY parties.name
 union
SELECT '總統/副總統' AS election_type,  
       parties.name AS party_name,
       round(SUM(presidents.votes)/(select sum(votes)*1.0 from presidents),4) AS votes_percentage
  FROM presidents
  JOIN candidates
    ON presidents.candidate_id = candidates.id
  JOIN parties
    ON candidates.party_id = parties.id
 GROUP BY parties.name;


"""