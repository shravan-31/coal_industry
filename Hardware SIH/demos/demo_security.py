from security import SecurityManager

def main():
    print("Demonstrating Security Manager for R&D Evaluation System")
    print("=" * 58)
    
    # Initialize security manager
    security = SecurityManager("demo_security.db")
    print("✓ Security Manager initialized")
    
    # Demonstrate security features
    print("\nSecurity Features Demo:")
    
    # 1. User authentication
    print("\n1. User Authentication:")
    user_info = security.authenticate_user("admin", "Admin123!")
    if user_info:
        print(f"   Authenticated: {user_info['username']} ({user_info['role']})")
    else:
        print("   Authentication failed")
    
    # 2. Session management
    print("\n2. Session Management:")
    session_id = security.create_session("admin", "127.0.0.1", "Demo Browser")
    print(f"   Session created: {session_id[:10]}...")
    
    session_info = security.validate_session(session_id)
    if session_info:
        print(f"   Session valid for: {session_info['username']}")
    else:
        print("   Session invalid")
    
    # 3. Permission checking
    print("\n3. Permission Checking:")
    has_permission = security.has_permission("admin", "all")
    print(f"   Admin has 'all' permission: {has_permission}")
    
    # 4. Data encryption
    print("\n4. Data Encryption:")
    sensitive_data = "This is sensitive proposal information that needs protection"
    encrypted = security.encrypt_data(sensitive_data)
    print(f"   Original: {sensitive_data[:30]}...")
    print(f"   Encrypted: {encrypted[:30]}...")
    
    decrypted = security.decrypt_data(encrypted)
    print(f"   Decrypted: {decrypted[:30]}...")
    print(f"   Match: {sensitive_data == decrypted}")
    
    # 5. Audit logging
    print("\n5. Audit Logging:")
    log_id = security.log_audit_event(
        user_id="admin",
        action="demo_access",
        resource="security_system",
        details={"demo": "test", "purpose": "demonstration"},
        ip_address="127.0.0.1",
        success=True
    )
    print(f"   Audit event logged (ID: {log_id})")
    
    # Get recent audit logs
    audit_logs = security.get_audit_log(limit=3)
    print(f"   Recent audit events: {len(audit_logs)}")
    if audit_logs:
        latest_log = audit_logs[0]
        print(f"   Latest: {latest_log['action']} by {latest_log['user_id']}")
    
    # 6. Create additional users for demonstration
    print("\n6. User Management:")
    
    # Create sample users if they don't exist
    sample_users = [
        ("demo001", "alice_researcher", "Alice123!", "reviewer", "alice@research.org"),
        ("demo002", "bob_submitter", "Bob123!", "submitter", "bob@company.com")
    ]
    
    for user_id, username, password, role, email in sample_users:
        if not security.user_exists(username):
            created = security.create_user(user_id, username, password, role, email)
            if created:
                print(f"   Created user: {username} ({role})")
            else:
                print(f"   Failed to create user: {username}")
        else:
            print(f"   User already exists: {username}")
    
    # Authenticate and test permissions
    print("\n7. Permission Testing:")
    reviewer_auth = security.authenticate_user("alice_researcher", "Alice123!")
    if reviewer_auth:
        print(f"   Authenticated: {reviewer_auth['username']}")
        
        # Test permissions
        can_evaluate = security.has_permission(reviewer_auth['user_id'], "evaluate")
        can_submit = security.has_permission(reviewer_auth['user_id'], "submit_proposals")
        print(f"   Can evaluate proposals: {can_evaluate}")
        print(f"   Can submit proposals: {can_submit}")
    
    submitter_auth = security.authenticate_user("bob_submitter", "Bob123!")
    if submitter_auth:
        print(f"   Authenticated: {submitter_auth['username']}")
        
        # Test permissions
        can_evaluate = security.has_permission(submitter_auth['user_id'], "evaluate")
        can_submit = security.has_permission(submitter_auth['user_id'], "submit_proposals")
        print(f"   Can evaluate proposals: {can_evaluate}")
        print(f"   Can submit proposals: {can_submit}")
    
    print("\n" + "=" * 58)
    print("✓ Security Manager demonstration completed successfully")
    print(f"\nNote: Demo security database saved as 'demo_security.db'")
    print("Note: Encryption key saved as 'encryption.key'")

if __name__ == "__main__":
    main()