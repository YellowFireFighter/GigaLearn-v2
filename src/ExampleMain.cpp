#include <GigaLearnCPP/Learner.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/Rewards/Reward.h>
#include "testrewards.h"
#include "RewardSchedule.h"
#include <atomic>

using namespace GGL;
using namespace RLGC;

static std::atomic<uint64_t> g_totalSteps{ 0 };

class SpeedTowardBallReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        Vec posDiff = state.ball.pos - player.pos;
        float dist = posDiff.Length();
        if (dist < 1.f) return 0.f;
        Vec dirToBall = posDiff / dist;
        float speedToward = player.vel.Dot(dirToBall);
        return RS_MAX(0.f, speedToward / CommonValues::CAR_MAX_SPEED);
    }
};

class VelocityBallToGoalOnTouchReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!player.ballTouchedStep) return 0.f;
        float goalY = (player.team == Team::ORANGE) ? -CommonValues::BACK_NET_Y : CommonValues::BACK_NET_Y;
        Vec posDiff = Vec(0, goalY, 0) - state.ball.pos;
        float dist = posDiff.Length();
        if (dist < 1.f) return 0.f;
        Vec dirToGoal = posDiff / dist;
        float velToward = state.ball.vel.Dot(dirToGoal);
        return RS_MAX(0.f, velToward / CommonValues::BALL_MAX_SPEED);
    }
};

class RandomStateSetter : public StateSetter {
public:
    KickoffState kickoff;
    RandomState random;

    RandomStateSetter() : kickoff(), random(true, true, false) {}

    void ResetArena(Arena* arena) override {
        if ((rand() % 10) < 8)
            kickoff.ResetArena(arena);
        else
            random.ResetArena(arena);
    }
};

EnvCreateResult EnvCreateFunc(int index) {
    std::vector<WeightedReward> rewards = BuildRewardsForStep(g_totalSteps.load());

    std::vector<TerminalCondition*> terminalConditions = {
        new NoTouchCondition(30),
        new GoalScoreCondition()
    };

    int playersPerTeam = 1;
    auto arena = Arena::Create(GameMode::SOCCAR);
    for (int i = 0; i < playersPerTeam; i++) {
        arena->AddCar(Team::BLUE);
        arena->AddCar(Team::ORANGE);
    }

    EnvCreateResult result = {};
    result.actionParser = new DefaultAction();
    result.obsBuilder = new AdvancedObs();
    result.stateSetter = new RandomStateSetter();
    result.terminalConditions = terminalConditions;
    result.rewards = rewards;
    result.arena = arena;

    return result;
}

int main(int argc, char* argv[]) {
    RocketSim::Init("collision_meshes");

    LearnerConfig cfg = {};
    cfg.deviceType = LearnerDeviceType::GPU_CUDA;
    cfg.tickSkip = 8;
    cfg.actionDelay = cfg.tickSkip - 1;

    cfg.numGames = 250;
    cfg.randomSeed = 123;

    int tsPerItr = 500'000;
    cfg.ppo.tsPerItr = tsPerItr;
    cfg.ppo.batchSize = tsPerItr;

    cfg.ppo.miniBatchSize = 100'000;

    cfg.ppo.epochs = 2;

    cfg.ppo.entropyScale = 0.035f;
    cfg.ppo.gaeGamma = 0.99;
    cfg.ppo.policyLR = 2e-4;
    cfg.ppo.criticLR = 2e-4;

    cfg.ppo.sharedHead.layerSizes = {};
    cfg.ppo.policy.layerSizes = { 1024, 1024, 1024, 1024, 1024, 512 };
    cfg.ppo.critic.layerSizes = { 1024, 1024, 1024, 512 };

    auto optim = ModelOptimType::ADAM;
    cfg.ppo.policy.optimType = optim;
    cfg.ppo.critic.optimType = optim;
    cfg.ppo.sharedHead.optimType = optim;

    auto activation = ModelActivationType::RELU;
    cfg.ppo.policy.activationType = activation;
    cfg.ppo.critic.activationType = activation;
    cfg.ppo.sharedHead.activationType = activation;

    bool addLayerNorm = true;
    cfg.ppo.policy.addLayerNorm = addLayerNorm;
    cfg.ppo.critic.addLayerNorm = addLayerNorm;
    cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;

    cfg.sendMetrics = true;
    cfg.metricsProjectName = "yxllowtechlarge";
    cfg.metricsGroupName = "bot";
    cfg.metricsRunName = "run4";
    cfg.renderMode = false;

    cfg.savePolicyVersions    = true;   // Keep old versions so 1.0B milestone can enable self-play
    cfg.tsPerVersion          = 25'000'000;
    cfg.maxOldVersions        = 32;
    cfg.trainAgainstOldVersions = false;  // Enabled automatically at 1.0B by MilestoneTracker
    cfg.trainAgainstOldChance   = 0.15f; 

    cfg.skillTracker.enabled = true;
    cfg.skillTracker.numArenas = 16;
    cfg.skillTracker.simTime = 45;
    cfg.skillTracker.maxSimTime = 240;
    cfg.skillTracker.updateInterval = 16;
    cfg.skillTracker.ratingInc = 5;

    bool renderMode = true;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--render") {
            cfg.sendMetrics = false;  
            cfg.ppo.deterministic = true; 
            cfg.renderMode = true;     
            renderMode = true;
            break;
        }
    }

    MilestoneTracker milestoneTracker;

    auto stepCallback = [&milestoneTracker](Learner* learner,
        const std::vector<RLGC::GameState>&,
        Report&) {
            g_totalSteps.store(learner->totalTimesteps);
            milestoneTracker.CheckAndApply(learner);
        };

    Learner* learner = new Learner(EnvCreateFunc, cfg, stepCallback);
    learner->Start();

    return EXIT_SUCCESS;
}