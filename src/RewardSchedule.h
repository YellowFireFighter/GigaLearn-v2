#pragma once
// RewardSchedule.h — milestone-based dynamic reward and training-parameter scheduler
//
// BuildRewardsForStep(totalSteps) returns a fresh, heap-allocated reward list that
//   matches the training plan for the given total step count.  Call it once per arena
//   (reward objects may NOT be shared across arenas).
//
// MilestoneTracker::CheckAndApply(learner) should be called inside the StepCallbackFn
//   passed to the Learner constructor.  It detects milestone crossings (including when
//   resuming from a checkpoint) and atomically rebuilds rewards + updates hyperparams.
//
// ── Training plan ────────────────────────────────────────────────────────────
//  0 – 100 M   TouchBall(5) | FaceBall(0.1) | VelPlayerToBall(1)
//  100 M+      + GoalReward(-0.80, 150) | + VelBallToGoal(5) | LR → 1e-4
//  550 M       entropyScale → 0.025
//  1.0 B       remove FaceBall | trainAgainstOldVersions = true
//  1.5 B       + KickoffProximity(1)
//  1.8 B       entropyScale → 0.02
//  2.0 B       TouchBall becomes ZeroSum
//  2.5 B       + PickupBoost(50)
// ─────────────────────────────────────────────────────────────────────────────

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/Rewards/Reward.h>
#include <GigaLearnCPP/Learner.h>

#include "testrewards.h"  // KickoffProximityReward

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace RLGC;
using namespace GGL;

// ─────────────────────────────────────────────────────────────────────────────
// BuildRewardsForStep
// Returns a fresh, heap-allocated reward list for the given total step count.
// Each call creates independent objects — call once per arena.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<WeightedReward> BuildRewardsForStep(uint64_t totalSteps) {
    const bool phase2        = totalSteps >= 100'000'000ULL;
    const bool removeFaceBall = totalSteps >= 1'000'000'000ULL;
    const bool hasKickoff    = totalSteps >= 1'500'000'000ULL;
    const bool zeroSumTouch  = totalSteps >= 2'000'000'000ULL;
    const bool hasPickupBoost = totalSteps >= 2'500'000'000ULL;

    std::vector<WeightedReward> rewards;

    // Phase 2+: GoalReward (already zero-sum internally)
    if (phase2)
        rewards.push_back({ new GoalReward(-0.80f), 150.f });

    // TouchBallReward — wrapped in ZeroSum at 2.0 B
    if (zeroSumTouch)
        rewards.push_back({ new ZeroSumReward(new TouchBallReward(), 1.f), 5.0f });
    else
        rewards.push_back({ new TouchBallReward(), 5.0f });

    // FaceBallReward removed at 1.0 B
    if (!removeFaceBall)
        rewards.push_back({ new FaceBallReward(), 0.1f });

    rewards.push_back({ new VelocityPlayerToBallReward(), 1.0f });

    // Phase 2+: VelocityBallToGoalReward
    if (phase2)
        rewards.push_back({ new VelocityBallToGoalReward(), 5.0f });

    // 1.5 B+: KickoffProximityReward
    if (hasKickoff)
        rewards.push_back({ new KickoffProximityReward(), 1.0f });

    // 2.5 B+: PickupBoostReward
    if (hasPickupBoost)
        rewards.push_back({ new PickupBoostReward(), 50.0f });

    return rewards;
}

// ─────────────────────────────────────────────────────────────────────────────
// MilestoneTracker
// Detects milestone crossings and updates rewards + hyperparameters.
// Instantiate once and pass CheckAndApply() as part of your StepCallbackFn.
// ─────────────────────────────────────────────────────────────────────────────
struct MilestoneTracker {

    // Ordered ascending — every entry must also have a matching case in
    // _applyConfig() so that config updates are cumulative and idempotent.
    static constexpr uint64_t MILESTONES[] = {
        100'000'000ULL,    // Phase 2 rewards | LR → 1e-4
        550'000'000ULL,    // entropyScale → 0.025
        1'000'000'000ULL,  // remove FaceBallReward | trainAgainstOldVersions
        1'500'000'000ULL,  // add KickoffProximityReward
        1'800'000'000ULL,  // entropyScale → 0.02
        2'000'000'000ULL,  // zero-sum TouchBallReward
        2'500'000'000ULL,  // add PickupBoostReward(50.f)
    };
    static constexpr const char* DESCRIPTIONS[] = {
        "Phase 2 rewards activated | LR → 1e-4",
        "entropyScale → 0.025",
        "FaceBallReward removed | trainAgainstOldVersions enabled",
        "KickoffProximityReward added",
        "entropyScale → 0.02",
        "TouchBallReward → ZeroSum",
        "PickupBoostReward(50) added",
    };
    static constexpr int NUM_MILESTONES =
        static_cast<int>(sizeof(MILESTONES) / sizeof(MILESTONES[0]));

    // Index of the next milestone that hasn't been applied yet
    int nextMilestoneIdx = 0;

    // Call this every step callback iteration.
    // Returns true if one or more milestones were newly applied.
    bool CheckAndApply(GGL::Learner* learner) {
        // All milestones already applied
        if (nextMilestoneIdx >= NUM_MILESTONES)
            return false;

        // Fast path: next milestone not yet reached
        if (learner->totalTimesteps < MILESTONES[nextMilestoneIdx])
            return false;

        // Advance past every newly reached milestone
        bool anyNew = false;
        while (nextMilestoneIdx < NUM_MILESTONES &&
               learner->totalTimesteps >= MILESTONES[nextMilestoneIdx]) {
            printf("[Schedule] %5lluM steps — %s\n",
                static_cast<unsigned long long>(MILESTONES[nextMilestoneIdx]) / 1'000'000ULL,
                DESCRIPTIONS[nextMilestoneIdx]);
            nextMilestoneIdx++;
            anyNew = true;
        }

        if (!anyNew)
            return false;

        // Apply cumulative hyperparameter config based on current step count
        _applyConfig(learner);

        // Rebuild rewards for every arena (once, reflecting current step count)
        //for (auto& arenaRewards : learner->envSet->rewards) {
        //    for (auto& wr : arenaRewards)
        //        delete wr.reward;
        //    arenaRewards = BuildRewardsForStep(learner->totalTimesteps);
        //}

        return true;
    }

private:
    void _applyConfig(GGL::Learner* learner) {
        const uint64_t steps = learner->totalTimesteps;

        // Learning rate: 1e-4 from 100 M
        if (steps >= 100'000'000ULL)
            learner->SetLearningRates(1e-4f, 1e-4f);

        // Entropy scale schedule
        if (steps >= 1'800'000'000ULL)
            learner->SetEntropyScale(0.02f);
        else if (steps >= 550'000'000ULL)
            learner->SetEntropyScale(0.025f);

        // Enable training against saved old versions at 1.0 B
        // (requires cfg.savePolicyVersions = true at startup so versionMgr exists)
        if (steps >= 1'000'000'000ULL)
            learner->config.trainAgainstOldVersions = true;
    }
};
