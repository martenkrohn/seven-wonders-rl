#include <iostream>
#include <cassert>
#include <player.h>
#include <card.h>
#include <wonder.h>
#include <resources.h>
#include <filer.h>

using namespace DMAG;

// Minimal cards isolate production without relying on a non-existent loader API.
std::vector<Card> ChoiceTestCards() {
    return {
        Card(CARD_ID::caravansery, "Caravansery", CARD_TYPE::commercial, 2,
             CARD_ID::none, std::vector<int>(8, 0), std::vector<int>(5, 1)),
        Card(CARD_ID::forum, "Forum", CARD_TYPE::commercial, 2,
             CARD_ID::none, std::vector<int>(8, 0), std::vector<int>(5, 1))
    };
}

// Helper to create a minimal test player
Player* CreateTestPlayer(Wonder* wonder) {
    Player* p = new Player();
    p->SetWonder(wonder);
    p->SetId(1);
    return p;
}

// Helper to create neighbors with no production
void SetupEmptyNeighbors(Player* p) {
    Player* east = new Player();
    Player* west = new Player();
    east->SetWonder(new Gizah_a());
    west->SetWonder(new Gizah_a());
    east->SetId(2);
    west->SetId(3);
    
    // Give them 0 coins so they can't sell anything
    east->AddResource(RESOURCE::coins, -3);
    west->AddResource(RESOURCE::coins, -3);
    
    p->SetNeighbors(east, west);
}

// Helper to build a card directly (simulating it was played)
void SimulateCardPlayed(Player* p, int card_id) {
    // We need to fake that the player has this card already built
    // The easiest way is to call BuildStructure with a proper card
    // But for testing, we'll manipulate internal state directly via IncrementOnDemand testing
    
    // Actually, let's just test IncrementOnDemand and ProduceResource directly
    // which is what matters for the bug fix
}

// Test 1: Caravansery cannot pay for two different raw resources in the same build
void Test_Caravansery_SingleUse() {
    std::cout << "\n=== Test 1: Caravansery can only provide ONE raw resource per turn ===\n";
    
    Player* player = CreateTestPlayer(new Gizah_a());
    SetupEmptyNeighbors(player);
    
    // Load all cards
    std::vector<Card> all_cards = ChoiceTestCards();
    Card caravansery_card;
    for (const auto& card : all_cards) {
        if (card.GetId() == CARD_ID::caravansery) {
            caravansery_card = card;
            break;
        }
    }
    
    // Give player enough coins to build caravansery
    player->AddResource(RESOURCE::coins, 10);
    
    // Build Caravansery
    std::vector<Card> hand = {caravansery_card};
    player->ReceiveCards(hand);
    bool built = player->BuildStructure(caravansery_card, hand, false);
    assert(built && "Test card must actually be built");
    std::cout << "  Built Caravansery: " << built << "\n";
    
    // Now test if Caravansery can be used twice in same turn
    player->ResetUsed();
    
    // First use: try to produce wood
    int wood_needed = player->IncrementOnDemand(RESOURCE::wood, 1, false);
    std::cout << "  After requesting wood, still needed: " << wood_needed << " (expected: 0)\n";
    
    // Second use: try to produce ore (should fail - Caravansery already used)
    int ore_needed = player->IncrementOnDemand(RESOURCE::ore, 1, false);
    std::cout << "  After requesting ore (same turn), still needed: " << ore_needed << " (expected: 1)\n";
    
    assert(wood_needed <= 0 && "Caravansery should provide first raw resource");
    assert(ore_needed > 0 && "Caravansery should NOT provide second raw resource in same turn");
    
    // Reset and try again - should work for fresh turn
    player->ResetUsed();
    int ore_needed_fresh = player->IncrementOnDemand(RESOURCE::ore, 1, false);
    std::cout << "  After reset, ore needed: " << ore_needed_fresh << " (expected: 0)\n";
    assert(ore_needed_fresh <= 0 && "After reset, Caravansery should work again");
    
    std::cout << "  ✓ PASS: Caravansery limited to one use per turn\n";
    
    delete player;
}

// Test 2: Forum cannot pay for multiple manufactured resources in the same build
void Test_Forum_SingleUse() {
    std::cout << "\n=== Test 2: Forum can only provide ONE manufactured resource per turn ===\n";
    
    Player* player = CreateTestPlayer(new Gizah_a());
    SetupEmptyNeighbors(player);
    
    // Load and build Forum
    std::vector<Card> all_cards = ChoiceTestCards();
    Card forum_card;
    for (const auto& card : all_cards) {
        if (card.GetId() == CARD_ID::forum) {
            forum_card = card;
            break;
        }
    }
    
    player->AddResource(RESOURCE::coins, 10);
    std::vector<Card> hand = {forum_card};
    player->ReceiveCards(hand);
    bool built = player->BuildStructure(forum_card, hand, false);
    assert(built && "Test card must actually be built");
    std::cout << "  Built Forum: " << built << "\n";
    
    // Test if Forum can be used twice
    player->ResetUsed();
    int loom_needed = player->IncrementOnDemand(RESOURCE::loom, 1, false);
    int glass_needed = player->IncrementOnDemand(RESOURCE::glass, 1, false);
    
    std::cout << "  After requesting loom, still needed: " << loom_needed << " (expected: 0)\n";
    std::cout << "  After requesting glass (same turn), still needed: " << glass_needed << " (expected: 1)\n";
    
    assert(loom_needed <= 0 && "Forum should provide first manufactured resource");
    assert(glass_needed > 0 && "Forum should NOT provide second manufactured resource in same turn");
    
    std::cout << "  ✓ PASS: Forum limited to one use per turn\n";
    
    delete player;
}

// Test 3: Alexandria choose-raw can't cover more than 1 raw per turn
void Test_Alexandria_Choice_SingleUse() {
    std::cout << "\n=== Test 3: Alexandria wonder choice can only provide ONE raw per turn ===\n";
    
    // Alexandria A stage 2 gives choose-raw ability
    Alexandria_a* alex = new Alexandria_a();
    Player* player = CreateTestPlayer(alex);
    SetupEmptyNeighbors(player);
    
    player->AddResource(RESOURCE::coins, 100); // Enough to build stages
    player->AddResource(RESOURCE::stone, 10);
    
    // Build first stage (costs stone:2)
    alex->AddStage(player);
    // Build second stage (costs ore:2) - this gives choose-raw
    player->AddResource(RESOURCE::ore, 2);
    alex->AddStage(player);
    
    std::cout << "  Alexandria A stage: " << alex->GetStage() << " (should be >= 2 for choose-raw)\n";
    
    // Now player has choose-raw ability, try to use it twice
    player->ResetUsed();
    bool can_produce_wood = player->ProduceResource(RESOURCE::wood, 1);
    bool can_produce_clay = player->ProduceResource(RESOURCE::clay, 1);
    
    std::cout << "  Can produce wood:1 = " << can_produce_wood << " (expected: true)\n";
    std::cout << "  Can produce clay:1 after wood = " << can_produce_clay << " (expected: false)\n";
    
    assert(can_produce_wood && "Alexandria should provide first raw choice");
    assert(!can_produce_clay && "Alexandria should NOT provide second raw choice in same turn");
    
    std::cout << "  ✓ PASS: Alexandria wonder choice limited to one use per turn\n";
    
    delete player;
}

// Test 4: Choice resets after end of turn
void Test_Choice_Reset() {
    std::cout << "\n=== Test 4: Choice tokens reset properly at turn boundaries ===\n";
    
    Player* player = CreateTestPlayer(new Gizah_a());
    SetupEmptyNeighbors(player);
    
    // Load and build Caravansery
    std::vector<Card> all_cards = ChoiceTestCards();
    Card caravansery_card;
    for (const auto& card : all_cards) {
        if (card.GetId() == CARD_ID::caravansery) {
            caravansery_card = card;
            break;
        }
    }
    
    player->AddResource(RESOURCE::coins, 10);
    std::vector<Card> hand = {caravansery_card};
    player->ReceiveCards(hand);
    player->BuildStructure(caravansery_card, hand, false);
    
    // Turn 1: Use caravansery for wood, then try ore
    player->ResetUsed();
    int turn1_wood = player->IncrementOnDemand(RESOURCE::wood, 1, false);
    int turn1_ore = player->IncrementOnDemand(RESOURCE::ore, 1, false);
    
    std::cout << "  Turn 1: wood_needed=" << turn1_wood << ", ore_needed=" << turn1_ore << "\n";
    assert(turn1_wood <= 0 && turn1_ore > 0 && "Within turn: second use should fail");
    
    // Turn 2: Reset and use again
    player->ResetUsed();
    int turn2_ore = player->IncrementOnDemand(RESOURCE::ore, 1, false);
    
    std::cout << "  Turn 2 (after reset): ore_needed=" << turn2_ore << "\n";
    assert(turn2_ore <= 0 && "After reset: should work again");
    
    std::cout << "  ✓ PASS: Choice tokens reset correctly\n";
    
    delete player;
}

// Test 5: Neighbor supply does not include Caravansery/Forum/Alexandria-choice
void Test_Neighbor_No_Choice() {
    std::cout << "\n=== Test 5: Neighbors cannot trade choice production ===\n";
    
    Player* player = CreateTestPlayer(new Gizah_a());
    Player* neighbor_east = new Player();
    Player* neighbor_west = new Player();
    
    neighbor_east->SetWonder(new Gizah_a());
    neighbor_west->SetWonder(new Gizah_a());
    neighbor_east->SetId(2);
    neighbor_west->SetId(3);
    
    player->SetNeighbors(neighbor_east, neighbor_west);
    
    // Load and build Caravansery for neighbor
    std::vector<Card> all_cards = ChoiceTestCards();
    Card caravansery_card;
    for (const auto& card : all_cards) {
        if (card.GetId() == CARD_ID::caravansery) {
            caravansery_card = card;
            break;
        }
    }
    
    neighbor_east->AddResource(RESOURCE::coins, 10);
    std::vector<Card> hand = {caravansery_card};
    neighbor_east->ReceiveCards(hand);
    neighbor_east->BuildStructure(caravansery_card, hand, false);
    
    // Player has coins but no wood production
    player->AddResource(RESOURCE::coins, 10);
    
    // Try to buy wood from neighbor (who only has Caravansery, no fixed wood)
    player->ResetUsed();
    neighbor_east->ResetUsed();
    
    // IncrementOnDemand with is_neighbor=true should NOT count Caravansery
    int neighbor_can_provide = neighbor_east->IncrementOnDemand(RESOURCE::wood, 1, true);
    assert(neighbor_can_provide == 0 && "Caravansery must not be traded");
    
    std::cout << "  Neighbor with Caravansery can provide wood: " << neighbor_can_provide << " (expected: > 0, should be just fixed production)\n";
    std::cout << "  (Note: neighbor has wonder production of stone, so wood should still be 0 from choice)\n";
    
    // The key test: neighbor with ONLY Caravansery (no fixed wood) should NOT provide wood
    // Since Gizah produces stone, not wood, neighbor_can_provide should be 0 or show only the wonder stone
    // Let's check it doesn't give us wood via choice
    
    // Give neighbor actual fixed wood production now
    neighbor_east->AddResource(RESOURCE::wood, 1);
    neighbor_east->ResetUsed();
    int neighbor_can_provide_with_fixed = neighbor_east->IncrementOnDemand(RESOURCE::wood, 1, true);
    
    std::cout << "  Neighbor with fixed wood can provide: " << neighbor_can_provide_with_fixed << " (expected: >= 1)\n";
    assert(neighbor_can_provide_with_fixed >= 1 && "Neighbor's fixed production should be tradeable");
    
    std::cout << "  ✓ PASS: Choice production not tradeable to neighbors\n";
    
    delete player;
    delete neighbor_east;
    delete neighbor_west;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Running Choice-Once-Per-Turn Test Suite\n";
    std::cout << "========================================\n";
    
    try {
        Test_Caravansery_SingleUse();
        Test_Forum_SingleUse();
        Test_Alexandria_Choice_SingleUse();
        Test_Choice_Reset();
        Test_Neighbor_No_Choice();
        
        std::cout << "\n========================================\n";
        std::cout << "✓ ALL TESTS PASSED\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n✗ TEST FAILED: Unknown error\n";
        return 1;
    }
}
